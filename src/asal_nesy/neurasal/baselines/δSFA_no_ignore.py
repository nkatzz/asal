#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeSyDFA — Neuro-Symbolic (Deterministic) Finite Automaton with predicate guards

This script trains a differentiable SFA/DFA from soft perceptual outputs without
enumerating predicate models. Transitions are guarded by small learnable DNFs
over a fixed predicate bank (even/odd, ≤k, ≥k). Row-softmax yields a stochastic
transition kernel per time step; a forward DP produces the acceptance
probability.

Key components
--------------
• PredicateBase      : per-step class probabilities → predicate soft-truths
• DNFEdge (softmax3) : differentiable DNF guard per edge (ternary literals)
• DNFEdge_GatePol    : differentiable DNF guard per edge (strength+polarity; IGNORE via s≈0)
• TransitionLayer    : predicate truths → row-stochastic transition per time step
• NeSyDFA            : forward dynamic program + trace variants
• Regularizers       : row entropy, selector sparsity, (optional) progress alignment,
                       flow concentration, and row partition (self vs forward)

Numerical safety
----------------
• bce_safe clamps NaNs/Infs and keeps predictions ∈ (0,1)
• TransitionLayer has a row-fallback: if a row would be all −inf, force a strong self-loop

Data
----
External data only: expects TensorSequence .pt files via --train-path and --test-path.
Each TensorSequence contains a symbolic digit sequence and a sequence-level label.
"""

import math, random, sys, argparse
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from δSFA_utils import count_parameters, count_by_child, build_transitions

# Optional pix guards (pix2rule-style AND/OR). Kept as-is; code here focuses on DNF.
try:
    from δSFA_neuralDNF import NeSyDFA_Pix  # pix2rule-based guards (AND/OR or hybrid)
    HAVE_PIX = True
except Exception:
    HAVE_PIX = False

# External data API (TensorSequence, loaders)
try:
    from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, TensorSequence
    HAVE_EXT = True
except Exception:
    HAVE_EXT = False

# Optional viz for extracted SFAs
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Repro + float32
np.random.seed(1); random.seed(1); torch.manual_seed(1)
torch.set_default_dtype(torch.float32)

# --------------------------------------------------------------------------
# Utility: symbolic digit → soft probability over 0..9 (mock CNN softmax)
# --------------------------------------------------------------------------
def cnn_probs(true_digit: int, sharp: float = 0.98) -> np.ndarray:
    """
    Create a peaked distribution at `true_digit` to emulate a CNN softmax.
    Used only when converting symbolic labels into softmax-like vectors.
    """
    probs = np.ones(10, np.float32) * (1 - sharp) / 9.0
    probs[true_digit] = sharp
    probs = probs * (1 - 0.01) + np.random.dirichlet(np.ones(10)).astype(np.float32) * 0.01
    probs /= probs.sum()
    return probs.astype(np.float32)

# --------------------------------------------------------------------------
# Base predicates u_j(t) from probabilities
# even, odd, le_0..le_9, ge_0..ge_9  (m = 22 predicates)
# --------------------------------------------------------------------------
class PredicateBase:
    """
    Deterministic mapping from per-step digit probabilities to predicate soft-truths.

    Predicates (total m = 22):
      • 0: even(d) = Σ_{d∈{0,2,4,6,8}} p(d)
      • 1: odd(d)  = Σ_{d∈{1,3,5,7,9}}  p(d)
      • 2..11: le_k(d) = P(d ≤ k) for k=0..9
      • 12..21: ge_k(d) = P(d ≥ k) for k=0..9
    """
    def __init__(self):
        self.even_idx = torch.tensor([0,2,4,6,8], dtype=torch.long)
        self.odd_idx  = torch.tensor([1,3,5,7,9], dtype=torch.long)

    def compute(self, digit_probs_t: torch.Tensor) -> torch.Tensor:
        """
        Map digit probs at a single time step to predicate soft-truths.
        digit_probs_t: (B,10) → (B,22)
        """
        digit_probs_t = digit_probs_t.float()
        even_truth = digit_probs_t.index_select(1, self.even_idx.to(digit_probs_t.device)).sum(1, keepdim=True)
        odd_truth  = digit_probs_t.index_select(1, self.odd_idx .to(digit_probs_t.device)).sum(1, keepdim=True)
        cumsum = torch.cumsum(digit_probs_t, dim=1)                         # (B,10)
        le_stack = torch.cat([cumsum[:, k:k+1] for k in range(10)], dim=1)  # (B,10)
        ones = torch.ones_like(cumsum[:, :1])
        ge_stack = torch.cat([ones, ones - cumsum[:, :-1]], dim=1)          # (B,10)
        return torch.cat([even_truth, odd_truth, le_stack, ge_stack], dim=1).float()

def predicate_names() -> List[str]:
    return ["even","odd"] + [f"le_{k}" for k in range(10)] + [f"ge_{k}" for k in range(10)]

def u_from_probs(bank: PredicateBase, prob_seq: torch.Tensor) -> torch.Tensor:
    """(B,T,10) → (B,T,22): stack PredicateBase.compute over time."""
    B, T, _ = prob_seq.size()
    return torch.stack([bank.compute(prob_seq[:, t, :]) for t in range(T)], dim=1)

# --------------------------------------------------------------------------
# TensorSequence → (B,T,10) probs and labels
# --------------------------------------------------------------------------
def _extract_digits_from_ts(ts: "TensorSequence") -> Tuple[List[int], int]:
    seq_len = ts.seq_length
    assert ts.dimensionality == 1, "Current SFA expects 1-D sequences (dimensionality==1)."
    digits = []
    for t in range(seq_len):
        digit_val = next(iter(ts.image_labels[t][0].values()))
        digits.append(int(digit_val))
    y = int(ts.seq_label)
    return digits, y

def _digits_to_probs(digits: List[int], sharp: float = 0.98) -> np.ndarray:
    return np.stack([cnn_probs(d, sharp=sharp) for d in digits], axis=0).astype(np.float32)

def batch_ts_to_xy(batch: List["TensorSequence"], sharp: float = 0.98) -> Tuple[torch.Tensor, torch.Tensor]:
    probs_list, labels_list = [], []
    for ts in batch:
        digits, y = _extract_digits_from_ts(ts)
        probs_list.append(_digits_to_probs(digits, sharp))
        labels_list.append(y)
    xb = torch.from_numpy(np.stack(probs_list, axis=0)).float()
    yb = torch.tensor(labels_list, dtype=torch.float32)
    return xb, yb

# --------------------------------------------------------------------------
# Numerically safe BCE
# --------------------------------------------------------------------------
def bce_safe(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with guardrails."""
    y_pred = torch.nan_to_num(y_pred, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6).clamp(1e-6, 1.0 - 1e-6)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1.0,       neginf=0.0).clamp(0.0, 1.0)
    return F.binary_cross_entropy(y_pred, y_true)

# --------------------------------------------------------------------------
# DNF guards (softmax3 = your current ternary literal head)
# --------------------------------------------------------------------------
class DNFEdge(nn.Module):
    """
    Differentiable DNF guard with ternary per-predicate literal selector:
      L = s_ign*1 + s_pos*u + s_neg*(1-u),  sel = softmax over {IGN,POS,NEG}.
      Clause:  C = Π_j L_j
      Guard:   g = 1 − Π_r (1 − α_r C_r),  α_r = σ(a_r)
    """
    def __init__(self, m: int, R: int = 2):
        super().__init__()
        self.m, self.R = m, R
        self.sel  = nn.Parameter(torch.zeros(R, m, 3))
        self.alog = nn.Parameter(torch.zeros(R))

    def forward(self, u_t: torch.Tensor, literal_temperature: float = 0.8) -> torch.Tensor:
        selector = F.softmax(self.sel / max(literal_temperature, 1e-6), dim=-1)  # (R,m,3)
        u = u_t.unsqueeze(1)                                                     # (B,1,m)
        L = (selector[..., 0].unsqueeze(0)
             + selector[..., 1].unsqueeze(0) * u
             + selector[..., 2].unsqueeze(0) * (1.0 - u))                        # (B,R,m)
        C = L.clamp(0.0, 1.0).prod(dim=-1)                                       # (B,R)
        alpha = torch.sigmoid(self.alog).unsqueeze(0)                             # (1,R)
        z = (alpha * C).clamp(0.0, 1.0)
        return 1.0 - (1.0 - z).prod(dim=-1)                                      # (B,)

    @torch.no_grad()
    def extract(self, names: List[str], ts: float = 0.75, ta: float = 0.2) -> List[List[Tuple[str, bool]]]:
        """Thresholded extraction for softmax3: keep literal if max{P(+),P(−)}≥ts; keep clause if α≥ta."""
        sel_p = F.softmax(self.sel, dim=-1).cpu().numpy()  # (R,m,3)
        alpha = torch.sigmoid(self.alog).cpu().numpy()     # (R,)
        dnfs: List[List[Tuple[str, bool]]] = []
        for r in range(self.R):
            if alpha[r] < ta: continue
            lits: List[Tuple[str, bool]] = []
            for j in range(self.m):
                pos, neg = sel_p[r, j, 1], sel_p[r, j, 2]
                if pos >= ts and pos >= neg: lits.append((names[j], True))
                elif neg >= ts and neg > pos: lits.append((names[j], False))
            if lits: dnfs.append(lits)
        return dnfs

# --------------------------------------------------------------------------
# NEW: DNF guards (gatepol = strength+polarity; IGNORE by s≈0)
# --------------------------------------------------------------------------
class DNFEdge_GatePol(nn.Module):
    """
    Differentiable DNF guard with per-literal strength and polarity:
      s = σ(a) ∈ [0,1] (IGNORE if s≈0),  p = σ(b) ∈ [0,1] (1→POS, 0→NEG)
      v = p*u + (1-p)*(1-u),  L = (1-s) + s*v
      Clause:  C = Π_j L_j
      Guard:   g = 1 − Π_r (1 − α_r C_r),  α_r = σ(a_r)
    """
    def __init__(self, m: int, R: int = 2):
        super().__init__()
        self.m, self.R = m, R
        self.a = nn.Parameter(torch.full((R, m), -1.5))  # strength logits (s≈0.18 init)
        self.b = nn.Parameter(torch.zeros(R, m))         # polarity logits (p≈0.5 init)
        self.alog = nn.Parameter(torch.zeros(R))         # clause logit

    def forward(self, u_t: torch.Tensor, literal_temperature: float = 0.8) -> torch.Tensor:
        s = torch.sigmoid(self.a)                      # (R,m)
        p = torch.sigmoid(self.b)                      # (R,m)
        u = u_t.unsqueeze(1)                           # (B,1,m)
        v = p * u + (1.0 - p) * (1.0 - u)              # (B,R,m)
        L = 1.0 - s * (1.0 - v)                        # (B,R,m)
        C = L.clamp(0.0, 1.0).prod(dim=-1)             # (B,R)
        alpha = torch.sigmoid(self.alog).unsqueeze(0)  # (1,R)
        z = (alpha * C).clamp(0.0, 1.0)
        return 1.0 - (1.0 - z).prod(dim=-1)            # (B,)

    @torch.no_grad()
    def extract(self, names: List[str], ts: float = 0.5, ta: float = 0.2, tp: float = 0.5) -> List[List[Tuple[str,bool]]]:
        """
        Thresholded extraction for gatepol:
          keep literal if s≥ts; polarity is POS if p≥tp else NEG; keep clause if α≥ta.
        """
        s = torch.sigmoid(self.a).cpu().numpy()        # (R,m)
        p = torch.sigmoid(self.b).cpu().numpy()        # (R,m)
        alpha = torch.sigmoid(self.alog).cpu().numpy() # (R,)
        dnfs: List[List[Tuple[str, bool]]] = []
        for r in range(self.R):
            if alpha[r] < ta: continue
            lits: List[Tuple[str, bool]] = []
            for j in range(self.m):
                if s[r, j] >= ts:
                    lits.append((names[j], bool(p[r, j] >= tp)))
            if lits: dnfs.append(lits)
        return dnfs

# --------------------------------------------------------------------------
# Transition layer
# --------------------------------------------------------------------------
class TransitionLayer(nn.Module):
    """
    Predicate truths → per-time row-stochastic transition via edge guards.
    Supports two literal parameterizations:
      • softmax3 (default): DNFEdge with {IGNORE,+,−} per predicate
      • gatepol : DNFEdge_GatePol with strength+polarity (IGNORE by s≈0)
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int, int]] = None,
                 literal_mode: str = "softmax3"):
        super().__init__()
        self.n, self.num_states = n, n
        self.num_predicates, self.max_clauses = m, R
        self.literal_mode = literal_mode

        EdgeClass = DNFEdge if literal_mode == "softmax3" else DNFEdge_GatePol
        self.edges = nn.ModuleList([EdgeClass(m, R) for _ in range(n * n)])

        mask = torch.full((n, n), float('-inf'))
        if allowed_edges is None:
            allowed_edges = [(q, q) for q in range(n)]
        for (src, dst) in allowed_edges:
            mask[src, dst] = 0.0
        self.register_buffer("mask", mask)

    def _row_fallback(self, logits: torch.Tensor) -> torch.Tensor:
        _, n, _ = logits.shape
        row_has_finite = torch.isfinite(logits).any(dim=-1, keepdim=True)  # (B,n,1)
        if not row_has_finite.all():
            eye = torch.eye(n, device=logits.device).unsqueeze(0)
            logits = torch.where(row_has_finite, logits, 10.0 * eye)
        return logits

    def forward(self, u_t: torch.Tensor, row_temperature: float = 0.5, literal_temperature: float = 0.8) -> torch.Tensor:
        B = u_t.size(0)
        g_list = [edge(u_t, literal_temperature) for edge in self.edges]   # list of (B,)
        G = torch.stack(g_list, dim=-1).view(B, self.n, self.n)            # (B,n,n)
        logits = torch.log(G.clamp_min(1e-6)) + self.mask
        logits = self._row_fallback(logits)
        return F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()

    def forward_with_g(self, u_t: torch.Tensor, row_temperature: float = 0.5, literal_temperature: float = 0.8):
        B = u_t.size(0)
        g_list = [edge(u_t, literal_temperature) for edge in self.edges]
        G = torch.stack(g_list, dim=-1).view(B, self.n, self.n)
        logits = torch.log(G.clamp_min(1e-6)) + self.mask
        logits = self._row_fallback(logits)
        M = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return M, G

    @torch.no_grad()
    def extract_all(self, names: List[str], ts: float = 0.75, ta: float = 0.2, tp: float = 0.5
                    ) -> Dict[Tuple[int,int], List[List[Tuple[str,bool]]]]:
        """
        Extract readable DNFs for all allowed edges.
        • softmax3: uses (ts, ta)
        • gatepol : uses (ts, ta, tp)
        """
        dnfs_by_edge: Dict[Tuple[int,int], List[List[Tuple[str,bool]]]] = {}
        for q in range(self.n):
            for qp in range(self.n):
                if not torch.isfinite(self.mask[q, qp]): continue
                e = self.edges[q * self.n + qp]
                if hasattr(e, "sel"):  # softmax3
                    clauses = e.extract(names, ts=ts, ta=ta)
                else:                   # gatepol
                    clauses = e.extract(names, ts=ts, ta=ta, tp=tp)
                dnfs_by_edge[(q+1, qp+1)] = clauses
        return dnfs_by_edge

# --------------------------------------------------------------------------
# DFA wrapper
# --------------------------------------------------------------------------
class NeSyDFA(nn.Module):
    """
    DFA wrapper: transition layer + forward dynamic program (+ optional traces).
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int, int]] = None,
                 accept_state: int = 3, literal_mode: str = "softmax3"):
        super().__init__()
        self.n, self.num_states = n, n
        self.transition = TransitionLayer(n, m, R, allowed_edges, literal_mode=literal_mode)
        self.register_buffer("alpha0", F.one_hot(torch.tensor(0), num_classes=n).float().unsqueeze(0))
        self.register_buffer("accept", F.one_hot(torch.tensor(accept_state), num_classes=n).float())

    def forward_with_traces(self, u_seq: torch.Tensor, row_temperature: float = 0.5, literal_temperature: float = 0.8):
        u_seq = u_seq.float()
        B, T, _ = u_seq.shape
        alpha = self.alpha0.repeat(B, 1)
        alphas = [alpha]
        Ms = []
        for t in range(T):
            M_t = self.transition(u_seq[:, t, :], row_temperature=row_temperature, literal_temperature=literal_temperature)
            Ms.append(M_t)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
            alphas.append(alpha)
        y_pred = (alpha * self.accept).sum(-1)
        return y_pred, alphas[:-1], Ms

    def forward_with_traces_and_g(self, u_seq: torch.Tensor, row_temperature: float = 0.5, literal_temperature: float = 0.8):
        u_seq = u_seq.float()
        B, T, _ = u_seq.shape
        alpha = self.alpha0.repeat(B, 1)
        alphas = [alpha]
        Ms, Gs = [], []
        for t in range(T):
            M_t, g_t = self.transition.forward_with_g(u_seq[:, t, :], row_temperature=row_temperature, literal_temperature=literal_temperature)
            Ms.append(M_t); Gs.append(g_t)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
            alphas.append(alpha)
        y_pred = (alpha * self.accept).sum(-1)
        return y_pred, alphas[:-1], Ms, Gs

    def forward(self, u_seq: torch.Tensor, row_temperature: float = 0.5, literal_temperature: float = 0.8) -> torch.Tensor:
        y_pred, _, _ = self.forward_with_traces(u_seq, row_temperature=row_temperature, literal_temperature=literal_temperature)
        return y_pred

# --------------------------------------------------------------------------
# Helpers for training losses (unchanged)
# --------------------------------------------------------------------------
def compute_intended_guards_from_u(u_seq: torch.Tensor):
    even = u_seq[:, :, 0]; odd = u_seq[:, :, 1]
    le5  = u_seq[:, :, 2 + 5]
    le2  = u_seq[:, :, 2 + 2]
    ge7  = u_seq[:, :, 2 + 10 + 7]
    G1 = even * ge7; G2 = odd * le5; G3 = le2
    return G1, G2, G3

def row_entropy(M: torch.Tensor) -> torch.Tensor:
    p = M.clamp_min(1e-8)
    return -(p * p.log()).sum(-1).mean()

def pretty_clause(clause: List[Tuple[str, bool]]) -> str:
    return " ∧ ".join([name if positive else f"NOT({name})" for name, positive in clause]) if clause else "TRUE"

def selector_sparsity_penalty(model) -> torch.Tensor:
    """
    Promote compact guards by penalizing *used* literals/clauses.

    • softmax3: mean(1 − P(IGNORE)) + mean(σ(a_r))
    • gatepol : mean(σ(a_strength)) + mean(σ(a_clause))
    • pix     : L1 on AND/OR weights (+ mean(σ(a_clause)) if hybrid)
    """
    penalty = torch.tensor(0.0, device=device); count = 0
    for edge in model.transition.edges:
        if hasattr(edge, "sel"):  # softmax3
            sel_p = F.softmax(edge.sel, dim=-1)
            penalty = penalty + (1.0 - sel_p[..., 0]).mean()
            if hasattr(edge, "alog"):
                penalty = penalty + torch.sigmoid(edge.alog).mean()
            count += 1
        elif hasattr(edge, "a") and hasattr(edge, "b"):  # gatepol
            penalty = penalty + torch.sigmoid(edge.a).mean()
            if hasattr(edge, "alog"):
                penalty = penalty + torch.sigmoid(edge.alog).mean()
            count += 1
        elif hasattr(edge, "and_units"):  # pix/hybrid
            for and_unit in edge.and_units:
                penalty = penalty + and_unit.w.abs().mean()
                count += 1
            if hasattr(edge, "or_unit"):
                penalty = penalty + edge.or_unit.w.abs().mean(); count += 1
            if hasattr(edge, "alog"):
                penalty = penalty + torch.sigmoid(edge.alog).mean(); count += 1
    if count == 0: return torch.tensor(0.0, device=device)
    return penalty / count

def rule_constraint_penalty(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft penalties to discourage contradictions/redundancy.
    Applies only to softmax3 (DNFEdge with 'sel').
    """
    p_contra = torch.tensor(0.0, device=device); p_am1 = torch.tensor(0.0, device=device)
    cnt_c = 0; cnt_m = 0
    idx_even, idx_odd = 0, 1
    idx_le = list(range(2, 12)); idx_ge = list(range(12, 22))
    mask_m_gt_k = torch.triu(torch.ones(10, 10, device=device), diagonal=1)
    for edge in model.transition.edges:
        if not hasattr(edge, "sel"): continue
        sel_probs = F.softmax(edge.sel, dim=-1)
        pos_ch = sel_probs[..., 1]
        pos_even = pos_ch[:, idx_even]; pos_odd = pos_ch[:, idx_odd]
        p_contra += (pos_even * pos_odd).mean(); cnt_c += 1
        le_pos = pos_ch[:, idx_le]; ge_pos = pos_ch[:, idx_ge]
        pair = le_pos.unsqueeze(-1) * ge_pos.unsqueeze(-2)
        p_contra += (pair * mask_m_gt_k).mean()
        excess_le = F.relu(le_pos.sum(-1) - 1.0).mean()
        excess_ge = F.relu(ge_pos.sum(-1) - 1.0).mean()
        p_am1 += (excess_le + excess_ge) * 0.5; cnt_m += 1
    if cnt_c > 0: p_contra = p_contra / cnt_c
    if cnt_m > 0: p_am1    = p_am1 / cnt_m
    return p_contra, p_am1

# --------------------------------------------------------------------------
# Evaluation helpers (unchanged)
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_f1_from_loader(model: NeSyDFA, bank: PredicateBase, loader: DataLoader,
                           tau_row: float = 0.3, tau_gs: float = 0.7, threshold: float = 0.5):
    model.eval(); TP = FP = TN = FN = 0
    for batch in loader:
        xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
        xb = xb.to(device); yb = yb.to(device)
        u_seq = u_from_probs(bank, xb)
        y_pred = model(u_seq, row_temperature=tau_row, literal_temperature=tau_gs)
        y_pred = torch.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        pred = (y_pred >= threshold).to(torch.int32)
        true = (yb >= 0.5).to(torch.int32)
        TP += int(((pred == 1) & (true == 1)).sum())
        FP += int(((pred == 1) & (true == 0)).sum())
        TN += int(((pred == 0) & (true == 0)).sum())
        FN += int(((pred == 0) & (true == 1)).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print("Confusion matrix (threshold = {:.2f}):".format(threshold))
    print(f"  TP={TP}  FP={FP}")
    print(f"  FN={FN}  TN={TN}")
    print("Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    return precision, recall, f1

@torch.no_grad()
def sweep_f1(model: NeSyDFA, bank: PredicateBase, loader: DataLoader,
             tau_row: float = 0.3, tau_gs: float = 0.7, steps: int = 101):
    scores, labels = [], []
    for batch in loader:
        xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
        xb = xb.to(device); yb = yb.to(device)
        u_seq = u_from_probs(bank, xb)
        y_pred = model(u_seq, row_temperature=tau_row, literal_temperature=tau_gs)
        scores.append(torch.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0).cpu())
        labels.append((yb >= 0.5).to(torch.int32).cpu())
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    def f1_at(th):
        pred = (scores >= th).astype(int)
        TP = int(((pred == 1) & (labels == 1)).sum())
        FP = int(((pred == 1) & (labels == 0)).sum())
        TN = int(((pred == 0) & (labels == 0)).sum())
        FN = int(((pred == 0) & (labels == 1)).sum())
        precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
        recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        return precision, recall, f1

    best_th, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0
    for i in range(steps):
        th = i / (steps - 1)
        p, r, f1 = f1_at(th)
        if f1 > best_f1:
            best_th, best_f1, best_p, best_r = th, f1, p, r
    print(f"Best F1 sweep → threshold={best_th:.2f}  F1={best_f1:.4f}  Precision={best_p:.4f}  Recall={best_r:.4f}")
    return best_th, best_f1, best_p, best_r

@torch.no_grad()
def summarize_edge_usage(model: NeSyDFA, bank: PredicateBase, loader: DataLoader,
                         tau_row: float = 0.4, tau_gs: float = 0.8, device: str = "cuda"):
    """
    Aggregate expected flow per edge to see which transitions are actually used.
    flow_t(q→q') = E[ α_t(q) · M_t(q,q') ], averaged across time/batches.
    """
    n = model.transition.n
    usage = torch.zeros(n, n, device=device)
    total_sequences = 0
    model.eval()
    for batch in loader:
        xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
        xb = xb.to(device)
        u_seq = u_from_probs(bank, xb)
        y_pred, state_beliefs, transition_matrices, _ = model.forward_with_traces_and_g(u_seq, tau_row, tau_gs)
        total_sequences += xb.size(0)
        for t in range(len(transition_matrices)):
            flow_t = (state_beliefs[t].unsqueeze(-1) * transition_matrices[t])  # (B,n,n)
            usage += flow_t.sum(0)
    usage /= max(1, total_sequences)
    pairs = [((i+1, j+1), float(usage[i, j].item())) for i in range(n) for j in range(n)]
    pairs.sort(key=lambda x: -x[1])
    print("Top edge usage:")
    for (q, qp), val in pairs[:10]:
        print(f"  {q}->{qp}: {val:.3f}")
    return usage

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _validate_allowed(allowed: List[Tuple[int, int]], num_states: int):
    out_deg = [0] * num_states
    for q, qp in allowed: out_deg[q] += 1
    bad = [q for q, d in enumerate(out_deg) if d == 0]
    if bad:
        raise ValueError(f"States with no outgoing edges: {bad}. Add at least a self-loop (q,q).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train δSFA with predicate guards.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--num-ors", type=int, default=1)

    # Regularizer strengths and temperatures
    parser.add_argument("--lambda-flow", type=float, default=5e-3)
    parser.add_argument("--lambda-part", type=float, default=1e-2)
    parser.add_argument("--part-xor",   type=float, default=1.0)
    parser.add_argument("--lambda-prog",type=float, default=5e-2)
    parser.add_argument("--tau-row",    type=float, default=0.4)  # 0.4  # 1.0

    parser.add_argument("--tau-gs",     type=float, default=1.0)  # 0.5  # Initially: 1.0

    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--test-path",  type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extract-th", type=float, default=0.20)  # ts: literal threshold
    parser.add_argument("--extract-ta", type=float, default=0.20)  # ta: clause α threshold
    parser.add_argument("--extract-tp", type=float, default=0.50,  # tp: polarity threshold (gatepol only)
                        help="Polarity threshold (p≥tp→positive) used only in --literal-mode gatepol.")

    parser.add_argument("--guard", choices=["dnf", "pix", "pix_hybrid"], default="dnf",
                        help="'dnf' = learn DNFs per edge; 'pix' = pix AND+OR; 'pix_hybrid' = pix AND + noisy-OR.")

    # NEW: literal parameterization for the DNF guards (affects only --guard dnf)
    parser.add_argument("--literal-mode", choices=["softmax3", "gatepol"], default="softmax3",
                        help="DNF literal parameterization: 'softmax3' (IGNORE,+,−) or 'gatepol' (strength+polarity).")

    # Row-temperature annealing (optional)
    parser.add_argument("--tau-row-start", type=float, default=1.0)  # initially: 1.0
    parser.add_argument("--tau-row-final", type=float, default=0.1)
    parser.add_argument("--tau-sched", choices=["linear", "cosine"], default="linear")

    # Any-time acceptance auxiliary loss (optional)
    parser.add_argument("--lambda-any", type=float, default=5e-3)

    parser.add_argument("--lambda-contradict", type=float, default=1e-3)
    parser.add_argument("--lambda-atmost-one", type=float, default=1e-3)

    args = parser.parse_args()

    allowed = build_transitions(args.num_states, allow_backward=True, allow_jumps=False)
    _validate_allowed(allowed, num_states=args.num_states)

    assert HAVE_EXT, "data_structs.get_data/get_data_loader not importable."
    train_data, test_data = get_data(args.train_path, args.test_path)
    train_loader = get_data_loader(train_data, args.batch_size, train=True)
    test_loader  = get_data_loader(test_data, args.batch_size,  train=False)
    print(f"[INFO] Loaded external TensorSequence data: {len(train_data)} train / {len(test_data)} test")

    base_predicates = PredicateBase()
    names = predicate_names()

    # Build model
    if args.guard == "dnf":
        model = NeSyDFA(n=args.num_states, m=22, R=args.num_ors,
                        allowed_edges=allowed, accept_state=args.num_states - 1,
                        literal_mode=args.literal_mode).to(device)
    else:
        assert HAVE_PIX, "NeSyDFA_Pix not found — make sure δSFA_neuralDNF.py is importable."
        use_or_unit = (args.guard == "pix")
        model = NeSyDFA_Pix(n=args.num_states, m=22, R=args.num_ors,
                            allowed_edges=allowed, accept_state=args.num_states - 1,
                            use_or_unit=use_or_unit, slope_and=1.5, slope_or=1.5).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = count_parameters(model, verbose=True)
    print(parameters, params); count_by_child(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Regularizers (as before)
    lambda_ent   = 1e-2
    lambda_sel   =  0.00002 # 1e-2
    if args.guard in ("pix", "pix_hybrid"):
        lambda_sel *= 0.05
    lambda_flow  = args.lambda_flow
    lambda_part  = args.lambda_part
    part_xor     = args.part_xor
    lambda_prog  = args.lambda_prog
    row_temperature_train   = args.tau_row
    literal_temperature_train = args.tau_gs
    row_temperature_eval    = 0.3
    literal_temperature_eval  = 0.7

    def eval_loss() -> float:
        model.eval(); total_loss, total_examples = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
                xb = xb.to(device); yb = yb.to(device)
                u_seq = u_from_probs(base_predicates, xb)
                y_pred = model(u_seq, row_temperature=row_temperature_eval, literal_temperature=literal_temperature_eval)
                total_loss += bce_safe(y_pred, yb).item() * xb.size(0)
                total_examples += xb.size(0)
        return total_loss / max(1, total_examples)

    print(f"[INFO] λ_flow={lambda_flow:.2e}  λ_part={lambda_part:.2e}  part_xor={part_xor:.2f}  "
          f"λ_prog={lambda_prog:.2e}  τ_row={row_temperature_train:.2f}  τ_gs={literal_temperature_train:.2f}")

    def _anneal_tau_row(epoch_idx: int, num_epochs_minus1: int, tau0: float, tau1: float, schedule: str = "linear") -> float:
        if schedule == "linear":
            s = epoch_idx / max(1, num_epochs_minus1)
            return tau0 + (tau1 - tau0) * s
        s = epoch_idx / max(1, num_epochs_minus1)
        return tau1 + 0.5 * (tau0 - tau1) * (1 + math.cos(math.pi * s))

    train_hist, val_hist = [], []
    for epoch in range(1, args.epochs + 1):
        row_temperature_train = _anneal_tau_row(epoch - 1, args.epochs - 1,
                                                args.tau_row_start, args.tau_row_final, args.tau_sched)
        model.train()
        epoch_loss_sum, epoch_examples = 0.0, 0

        for batch in train_loader:
            xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
            xb = xb.to(device); yb = yb.to(device)
            u_seq = u_from_probs(base_predicates, xb)

            y_pred, state_beliefs, transition_matrices, guard_strengths = \
                model.forward_with_traces_and_g(u_seq, row_temperature=row_temperature_train,
                                                literal_temperature=literal_temperature_train)

            loss_bce = bce_safe(y_pred, yb)

            # Optional auxiliaries (kept disabled as in your current script)
            # loss_row_ent = torch.stack([row_entropy(M_t) for M_t in transition_matrices]).mean()

            loss_sel = selector_sparsity_penalty(model)

            # Flow / partition kept off by default (as in your script)
            # flow_terms = []
            # for t, M_t in enumerate(transition_matrices):
            #     flow_t = state_beliefs[t].unsqueeze(-1) * M_t
            #     flow_terms.append(-(flow_t ** 2).mean())
            # loss_flow = torch.stack(flow_terms).mean()

            part_terms = []
            for t, g_raw in enumerate(guard_strengths):
                for q in range(model.n - 1):
                    if torch.isfinite(model.transition.mask[q, q]) and torch.isfinite(model.transition.mask[q, q+1]):
                        g_self = g_raw[:, q, q]; g_fwd = g_raw[:, q, q+1]
                        part_terms.append(((g_self + g_fwd - 1.0)**2 + part_xor * (g_self * g_fwd)).mean())
            loss_part = torch.stack(part_terms).mean() if part_terms else torch.tensor(0.0, device=device)

            # Total loss (mirrors your current on-by-default terms)
            loss = (loss_bce
                    + lambda_sel   * loss_sel
                    # + lambda_flow  * loss_flow
                    + lambda_part  * loss_part
                    )

            # Optional rule constraints (disabled in your script)
            # if args.lambda_contradict > 0.0 or args.lambda_atmost_one > 0.0:
            #     p_contra, p_am1 = rule_constraint_penalty(model)
            #     loss = loss + args.lambda_contradict * p_contra + args.lambda_atmost_one * p_am1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * xb.size(0)
            epoch_examples += xb.size(0)

        train_hist.append(epoch_loss_sum / epoch_examples)
        val_loss = eval_loss(); val_hist.append(val_loss)
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_hist[-1]:.4f}  val={val_loss:.4f}")

    # Curves
    plt.figure()
    plt.plot(train_hist, label="train"); plt.plot(val_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.title("Training/Validation")
    plt.tight_layout(); plt.savefig("loss_curve.png")

    # Edge usage summary on train set
    summarize_edge_usage(model, base_predicates, train_loader)

    # ---------------------------------------------
    # Extraction thresholds reminder:
    #   ts  (literal threshold)
    #       • softmax3 (DNF): threshold on the literal *selector* softmax per predicate.
    #         A literal is included in a clause only if max{P(+),P(−)} ≥ ts.
    #       • gatepol : threshold on the *strength* s=σ(a). Keep literal if s≥ts.
    #
    #   tp  (polarity threshold; gatepol only)
    #       • If p≥tp → positive; else negative.
    #
    #   ta  (clause activation threshold)
    #       • threshold on the clause strength α=σ(alog). Keep clause if α ≥ ta.
    # ---------------------------------------------
    dnfs_by_edge = model.transition.extract_all(names, ts=args.extract_th, ta=args.extract_ta, tp=args.extract_tp)
    print("\nLearned guards (DNFs) per ALLOWED edge (q->q'):\n")
    for (q, qp), clauses in dnfs_by_edge.items():
        if not clauses: continue
        print(f"{q}->{qp}:  " + "  OR  ".join(pretty_clause(c) for c in clauses))

    # Optional graph viz
    if HAVE_NX:
        G = nx.DiGraph()
        for s in range(1, args.num_states + 1):
            G.add_node(s, label=f"{s}" + (" (accept)" if s == args.num_states else ""))
        for (q, qp), clauses in dnfs_by_edge.items():
            if not clauses: continue
            G.add_edge(q, qp, label=" OR ".join(pretty_clause(c) for c in clauses))
        pos = nx.spring_layout(G, seed=2)
        plt.figure()
        nx.draw_networkx_nodes(G, pos); nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=8)
        plt.title("Extracted SFA"); plt.axis('off'); plt.tight_layout()
        plt.savefig("extracted_sfa.png")

    print("\nSaved:")
    print(" - loss_curve.png")
    if HAVE_NX: print(" - extracted_sfa.png")

    # Evaluation on test set
    test_loader = get_data_loader(test_data, batch_size=512, train=False)
    compute_f1_from_loader(model, base_predicates, test_loader, tau_row=0.3, tau_gs=0.7, threshold=0.5)
    sweep_f1(model, base_predicates, test_loader, tau_row=0.3, tau_gs=0.7, steps=101)
