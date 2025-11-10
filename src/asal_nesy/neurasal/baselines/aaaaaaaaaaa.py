#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeSyDFA — Neuro-Symbolic (Deterministic) Finite Automaton with predicate guards

Multivariate extension (this version)
-------------------------------------
• Supports D>1 dimensions per time step (e.g., 2 digits per step for 2-variate MNIST).
• Predicate bank is applied **per dimension** and concatenated:
    u_t = [u_t^(1) || u_t^(2) || ... || u_t^(D)]  ∈ R^{22·D}
• Predicate names include dimension prefixes (e.g., d2_ge_7), used only for extraction/printing.

Unchanged core
--------------
• Guard parameterization (DNF), transition row-softmax, forward DP, losses, evaluation, extraction.
• CLI and default behavior (works for univariate data without any changes).

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
# from δSFA_utils import count_parameters, count_by_child, build_transitions
import δSFA_utils as utils

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

# Global active-predicate mask (indices into last dim of u-seq). If not None,
# u_from_probs_any() will slice its output columns to this selection.
from typing import Optional
ACTIVE_PRED_IDX = None  # type: Optional[torch.LongTensor]







# --------------------------------------------------------------------------
# Utility: symbolic digit → soft probability over 0..9 (mock CNN softmax)
# --------------------------------------------------------------------------
def cnn_probs(true_digit: int, sharp: float = 0.98) -> np.ndarray:
    """
    Create a peaked distribution at `true_digit` to emulate a CNN softmax.

    Args
    ----
    true_digit : int
        Ground-truth digit ∈ {0,...,9}.
    sharp : float
        Probability mass placed on the true class (rest is uniform).

    Returns
    -------
    probs : (10,) np.ndarray (float32)
        A probability vector that sums to 1.

    Notes
    -----
    • Used only when converting symbolic labels into softmax-like vectors.
    • Replace with a *real* CNN softmax if you connect a vision front-end.
    """
    probs = np.ones(10, np.float32) * (1 - sharp) / 9.0
    probs[true_digit] = sharp
    # tiny noise (Dirichlet) to avoid exact zeros/ones
    probs = probs * (1 - 0.01) + np.random.dirichlet(np.ones(10)).astype(np.float32) * 0.01
    probs /= probs.sum()
    return probs.astype(np.float32)

# --------------------------------------------------------------------------
# Base predicates u_j(t) from probabilities
# even, odd, le_0..le_9, ge_0..ge_9  (m = 22 predicates per dimension)
# --------------------------------------------------------------------------
class PredicateBase:
    """
    Deterministic mapping from per-step digit probabilities to predicate soft-truths.

    Predicates (total m = 22):
      • 0: even(d) = Σ_{d∈{0,2,4,6,8}} p(d)
      • 1: odd(d)  = Σ_{d∈{1,3,5,7,9}}  p(d)
      • 2..11: le_k(d) = P(d ≤ k) for k=0..9         (cumulative from left)
      • 12..21: ge_k(d) = P(d ≥ k) for k=0..9        (cumulative from right)

    Methods
    -------
    compute(p_t):
        Given (B,10) probabilities at time t, return (B,22) predicate truths, all in [0,1].
    """
    def __init__(self):
        self.even_idx = torch.tensor([0,2,4,6,8], dtype=torch.long)
        self.odd_idx  = torch.tensor([1,3,5,7,9], dtype=torch.long)

    def compute(self, digit_probs_t: torch.Tensor) -> torch.Tensor:
        """
        Map digit probs at a single time step to predicate soft-truths.

        Args
        ----
        digit_probs_t : (batch_size, 10) float32
            Per-sequence probabilities over digits 0..9 at time t.

        Returns
        -------
        predicate_truths_t : (batch_size, 22) float32
            Concatenation of [even, odd, le_0..le_9, ge_0..ge_9].
        """
        digit_probs_t = digit_probs_t.float()
        even_truth = digit_probs_t.index_select(1, self.even_idx.to(digit_probs_t.device)).sum(1, keepdim=True)
        odd_truth  = digit_probs_t.index_select(1, self.odd_idx .to(digit_probs_t.device)).sum(1, keepdim=True)

        # cumulative (left-to-right) for le_k
        cumsum = torch.cumsum(digit_probs_t, dim=1)                         # (B,10)
        le_stack = torch.cat([cumsum[:, k:k+1] for k in range(10)], dim=1)  # (B,10)

        # cumulative (right-to-left) for ge_k
        ones = torch.ones_like(cumsum[:, :1])
        ge_stack = torch.cat([ones, ones - cumsum[:, :-1]], dim=1)          # (B,10)

        predicate_truths_t = torch.cat([even_truth, odd_truth, le_stack, ge_stack], dim=1).float()
        return predicate_truths_t

def _base_predicate_names() -> List[str]:
    """Unprefixed names for the 22 base predicates in `PredicateBase`."""
    return ["even","odd"] + [f"le_{k}" for k in range(10)] + [f"ge_{k}" for k in range(10)]

def predicate_names_multi(dim_names: List[str], D: int) -> List[str]:
    """
    Build human-readable names for multivariate predicates.

    Example (D=2, dim_names=['d1','d2']):
      ['d1_even','d1_odd','d1_le_0',...,'d1_ge_9',  'd2_even','d2_odd',...,'d2_ge_9']
    """
    base = _base_predicate_names()
    res = []
    for d in range(D):
        prefix = dim_names[d] if d < len(dim_names) else f"d{d+1}"
        res.extend([f"{prefix}_{n}" for n in base])
    return res





def _normalize_pred_token(tok: str) -> tuple[str, str]:
    """
    Accept both 'd1_even' and 'even_d1'. Return ('d1', 'even').
    """
    tok = tok.strip()
    if not tok:
        raise ValueError("Empty predicate token")
    parts = tok.split('_')
    if len(parts) < 2:
        raise ValueError(f"Bad predicate token: {tok}")
    # try prefix form: dK_...
    if parts[0].startswith('d') and parts[0][1:].isdigit():
        dim = parts[0]
        base = '_'.join(parts[1:])
        return dim, base
    # try suffix form: ..._dK
    if parts[-1].startswith('d') and parts[-1][1:].isdigit():
        dim = parts[-1]
        base = '_'.join(parts[:-1])
        return dim, base
    # default: assume prefix form already
    return parts[0], '_'.join(parts[1:])

def _build_keep_indices(all_names: list[str], keep_list: list[str]) -> list[int]:
    """
    Map a list of tokens into indices of all_names. Tokens can be 'd1_even' or 'even_d1'.
    """
    # Build a lookup that accepts both 'd1_even' and 'even_d1'
    name_set = set(all_names)
    indices = []
    for tok in keep_list:
        d, base = _normalize_pred_token(tok)
        canonical = f"{d}_{base}"         # our script uses 'd1_even' style
        alt       = f"{base}_{d}"         # accept 'even_d1' too
        if canonical in name_set:
            indices.append(all_names.index(canonical))
        elif alt in name_set:
            indices.append(all_names.index(alt))
        else:
            raise ValueError(f"Predicate token '{tok}' not found in available names.")
    # de-duplicate while preserving order
    seen = set()
    out = []
    for i in indices:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

def _build_drop_indices(all_names: list[str], drop_list: list[str]) -> list[int]:
    """
    Produce the complement of a drop list.
    """
    to_drop = set(_build_keep_indices(all_names, drop_list))
    return [i for i in range(len(all_names)) if i not in to_drop]






# Global at top of file (already suggested earlier)
# ACTIVE_PRED_IDX: torch.Tensor | None = None

def u_from_probs_any(bank: PredicateBase, prob_seq: torch.Tensor) -> torch.Tensor:
    """
    Convert a sequence of digit probabilities into per-time predicate truths.

    Shape-agnostic:
      • (B,T,10)      → (B,T,22)
      • (B,T,D,10)    → (B,T,22·D) via per-dimension compute() and concatenation.

    If ACTIVE_PRED_IDX is set (LongTensor of selected predicate indices),
    the output's last dimension is sliced accordingly: (B,T,|ACTIVE_PRED_IDX|).
    """
    if prob_seq.dim() == 3:
        # Univariate: (B,T,10)
        B, T, _ = prob_seq.size()
        out = torch.stack([bank.compute(prob_seq[:, t, :]) for t in range(T)], dim=1)  # (B,T,22)

    elif prob_seq.dim() == 4:
        # Multivariate: (B,T,D,10) → concat per-dim predicates → (B,T,22·D)
        B, T, D, _ = prob_seq.size()
        per_t = []
        for t in range(T):
            per_d = []
            for d in range(D):
                per_d.append(bank.compute(prob_seq[:, t, d, :]))  # (B,22)
            per_t.append(torch.cat(per_d, dim=1))                  # (B,22·D)
        out = torch.stack(per_t, dim=1)                            # (B,T,22·D)

    else:
        raise ValueError(f"u_from_probs_any: unexpected prob_seq shape {tuple(prob_seq.shape)}")

    # Optional predicate gating (apply once, centrally)
    global ACTIVE_PRED_IDX
    if ACTIVE_PRED_IDX is not None:
        idx = ACTIVE_PRED_IDX.to(out.device)                       # (K,)
        out = out.index_select(dim=-1, index=idx)                  # (B,T,K)

    return out


# --------------------------------------------------------------------------
# TensorSequence utilities: read symbolic digits & labels and make (B,T,D,10)
# --------------------------------------------------------------------------
def _infer_dim_names(ts: "TensorSequence") -> List[str]:
    """
    Infer dimension (attribute) names from the first time step of a TensorSequence.

    We expect each `image_labels[t][d]` to be a dict with (typically) one key
    (e.g., {'d1': 7}). If multiple keys exist, we arbitrarily pick the first
    in each dict to build names; if empty or ambiguous, we fall back to d1..dD.
    """
    D = ts.dimensionality
    names = []
    for d in range(D):
        keys = list(ts.image_labels[0][d].keys())
        names.append(keys[0] if keys else f"d{d+1}")
    return names

def _extract_digits_grid(ts: "TensorSequence") -> Tuple[np.ndarray, int, List[str]]:
    """
    Read the symbolic digit grid and label from a (possibly multivariate) TensorSequence.

    Returns
    -------
    digits : (T,D) int numpy array
        The per-time, per-dimension symbolic digits.
    y      : int
        The sequence label in {0,1}.
    dim_names : List[str]
        Attribute names per dimension (e.g., ['d1','d2',...]).
    """
    T = ts.seq_length
    D = ts.dimensionality
    dim_names = _infer_dim_names(ts)
    grid = np.zeros((T, D), dtype=np.int32)
    for t in range(T):
        for d in range(D):
            # take the first key/value from the dict at (t,d)
            val = next(iter(ts.image_labels[t][d].values()))
            grid[t, d] = int(val)
    y = int(ts.seq_label)
    return grid, y, dim_names

def _digits_to_probs_grid(digits_TD: np.ndarray, cnn_out_size, sharp: float = 0.98) -> np.ndarray:
    """
    Vectorize a whole (T,D) grid of symbolic digits into softmax-like probs.

    Returns
    -------
    probs : (T,D,10) float32
    """
    T, D = digits_TD.shape
    out = np.empty((T, D, 10), dtype=np.float32)
    for t in range(T):
        for d in range(D):
            out[t, d, :] = cnn_probs(int(digits_TD[t, d]), sharp=sharp)
    return out

def batch_ts_to_xy(batch: List["TensorSequence"], sharp: float = 0.98) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Convert a list of TensorSequence to tensors usable by the model.

    Returns
    -------
    xb : (batch_size, seq_len, D, 10) float32
        Per-time, per-dimension digit probabilities. (For univariate, D=1.)
    yb : (batch_size,) float32
        Sequence labels in {0,1}.
    dim_names : List[str]
        Names per dimension, inferred from the first sequence in the batch.
        (Used only to build readable predicate names.)
    """
    assert len(batch) > 0, "Empty batch"
    # Use the first sequence to infer D and names; we assume batch is homogeneous
    first_grid, first_y, dim_names = _extract_digits_grid(batch[0])
    T, D = first_grid.shape

    probs_list, labels_list = [], []
    # process the first (already parsed) sequence
    probs_list.append(_digits_to_probs_grid(first_grid, sharp))
    labels_list.append(first_y)
    # process the rest
    for ts in batch[1:]:
        grid, y, _ = _extract_digits_grid(ts)
        assert grid.shape == (T, D), "All sequences in a batch must share (T,D)"
        probs_list.append(_digits_to_probs_grid(grid, sharp))
        labels_list.append(y)

    xb = torch.from_numpy(np.stack(probs_list, axis=0)).float()  # (B,T,D,10)
    yb = torch.tensor(labels_list, dtype=torch.float32)
    return xb, yb, dim_names

# --------------------------------------------------------------------------
# Numerically safe BCE
# --------------------------------------------------------------------------
def bce_safe(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Stable binary cross-entropy that clamps NaN/Inf and bounds probabilities.
    """
    y_pred = torch.nan_to_num(y_pred, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6).clamp(1e-6, 1.0 - 1e-6)
    y_true = torch.nan_to_num(y_true, nan=0.0, posinf=1.0,       neginf=0.0).clamp(0.0, 1.0)
    return F.binary_cross_entropy(y_pred, y_true)

# --------------------------------------------------------------------------
# DNF guard per edge (ternary literals) + Transition layer
# --------------------------------------------------------------------------
class DNFEdge(nn.Module):
    """
    Differentiable DNF guard g(x_t) for a single directed edge (state q → state q').

    Literals per predicate j∈{1..m} are chosen from {IGNORE, +, −} via a softmax.
    Clause r is a product over literals; clauses combine via noisy-OR.
    """
    def __init__(self, m: int, R: int = 2):
        super().__init__()
        self.m = m
        self.R = R
        self.sel  = nn.Parameter(torch.zeros(R, m, 3))  # selector logits
        self.alog = nn.Parameter(torch.zeros(R))        # clause strength logits

    """
    def forward(self, predicate_truths_t: torch.Tensor, literal_temperature: float = 0.8) -> torch.Tensor:
        u_t = predicate_truths_t.float()                                # (B,m)
        selector = F.softmax(self.sel / max(literal_temperature, 1e-6), dim=-1)  # (R,m,3)
        # L = s_ignore + s_pos*u + s_neg*(1-u)
        u = u_t.unsqueeze(1)                                            # (B,1,m)
        literal_vals = (selector[..., 0].unsqueeze(0)
                        + selector[..., 1].unsqueeze(0) * u
                        + selector[..., 2].unsqueeze(0) * (1.0 - u))    # (B,R,m)
        clause_vals = torch.prod(literal_vals, dim=-1)                  # (B,R)
        alpha = torch.sigmoid(self.alog).unsqueeze(0)                   # (1,R)
        g_t = 1.0 - torch.prod(1.0 - alpha * clause_vals, dim=-1)       # (B,)
        return g_t.float()
    """
    # """
    def forward(self, predicate_truths_t: torch.Tensor, literal_temperature: float = 0.8) -> torch.Tensor:
        
        # Compute g_t(q→q') for a batch at a single time step.
        #
        # Clause r:
        #   • Selector softmax over {IGNORE, +, −} per predicate.
        #   • Only +/− contribute; IGN contributes 0 evidence.
        #   • Use signed evidence with logit(u):  +logit(u) for POS, −logit(u) for NEG.
        #   • Length-invariant averaging over active literals (prevents collapse with large m).
        #   • IMPORTANT: empty clause (no active literals) evaluates to FALSE (0).
        #   • Clause value in [0,1] via sigmoid of averaged evidence.
        
        u_t = predicate_truths_t.float()  # (B,m)
        B, m = u_t.shape
        eps = 1e-6

        # Selector over {IGN, +, −}
        sel = F.softmax(self.sel / max(literal_temperature, 1e-6), dim=-1)  # (R,m,3)

        # Clamp predicate truths and compute logits
        # torch.logit(x, eps) clamps internally to [eps, 1-eps] for stability
        u = u_t.unsqueeze(1)  # (B,1,m)
        u_logit = torch.logit(u, eps=eps)  # (B,1,m)

        # Split channels; shape (1,R,m) so they broadcast over batch
        w_pos = sel[..., 1].unsqueeze(0)  # (1,R,m)
        w_neg = sel[..., 2].unsqueeze(0)  # (1,R,m)
        # s_act = w_pos + w_neg  # sum only over active literals, active mass (exclude IGN), (1,R,m)

        log_and = (w_pos * torch.log(u + eps) + w_neg * torch.log(1.0 - u + eps))  # (B,R,m)

        s_act = w_pos + w_neg # sum only over active literals/active mass (exclude IGN), (1,R,m)
        log_clause = (log_and * s_act).sum(dim=-1) / (s_act.sum(dim=-1) + eps)  # optional /avg; or drop the division
        clause_vals = torch.exp(log_clause).clamp(0.0, 1.0)  # (B,R)
        # keep empty-clause = 0:
        is_empty = (s_act.sum(dim=-1) <= 1e-12)
        clause_vals = torch.where(is_empty, torch.zeros_like(clause_vals), clause_vals)
        # combine clauses (keep noisy-OR or restore α if you like)
        g_t = 1.0 - torch.prod(1.0 - clause_vals, dim=-1)

        """
        # Signed evidence per predicate:
        #   e_j = (w_pos - w_neg) * logit(u_j) ; IGN contributes 0
        e = (w_pos - w_neg) * u_logit  # (B,R,m)
        e = torch.where(s_act > 0, e, torch.zeros_like(e))  # ignore preds → 0 evidence

        # Length-invariant averaging over *active* literals
        Z_raw = s_act.sum(dim=-1, keepdim=True)  # (1,R,1)  sum of active weights
        is_empty = (Z_raw <= 1e-12)  # (1,R,1) boolean for empty clause
        Z_safe = torch.where(is_empty, torch.ones_like(Z_raw), Z_raw)  # avoid /0
        a = s_act / (Z_safe + eps)  # (1,R,m) attention over active lits
        a = torch.where(is_empty, torch.zeros_like(a), a)  # zero attention for empty clauses

        clause_evidence = (a * e).sum(dim=-1)  # (B,R) averaged signed evidence
        clause_vals = torch.sigmoid(clause_evidence)  # (B,R) in [0,1]

        # Empty clause ⇒ FALSE (0)
        is_empty_BR = is_empty.squeeze(-1).expand(clause_vals.size(0), -1)  # (B,R)
        clause_vals = torch.where(is_empty_BR, torch.zeros_like(clause_vals), clause_vals)

        # Noisy-OR across clauses with learnable strengths α_r = σ(a_r)
        alpha = torch.sigmoid(self.alog).unsqueeze(0)  # (1,R)
        g_t = 1.0 - torch.prod(1.0 - alpha * clause_vals, dim=-1)  # (B,)
        # g_t = 1.0 - torch.prod(1.0 - clause_vals, dim=-1)
        """
        return g_t.float()
   # """

    @torch.no_grad()
    def extract(self, names: List[str], ts: float = 0.75, ta: float = 0.2) -> List[List[Tuple[str, bool]]]:
        selector_probs = F.softmax(self.sel, dim=-1).cpu().numpy()  # (R,m,3)
        alpha = torch.sigmoid(self.alog).cpu().numpy()              # (R,)
        dnfs: List[List[Tuple[str, bool]]] = []
        for r in range(self.R):
            # if alpha[r] < ta:
            #    continue
            clause_lits: List[Tuple[str, bool]] = []
            for j in range(self.m):
                pos, neg = selector_probs[r, j, 1], selector_probs[r, j, 2]
                if pos >= ts and pos >= neg:
                    clause_lits.append((names[j], True))
                elif neg >= ts and neg > pos:
                    clause_lits.append((names[j], False))
            if clause_lits:
                dnfs.append(clause_lits)
        return dnfs

class TransitionLayer(nn.Module):
    """
    Predicate truths → per-time **row-stochastic** transition kernel via edge guards.
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int, int]] = None):
        super().__init__()
        self.n = n
        self.num_states = n
        self.num_predicates = m
        self.max_clauses = R
        self.edges = nn.ModuleList([DNFEdge(m, R) for _ in range(n * n)])
        mask = torch.full((n, n), float('-inf'))
        if allowed_edges is None:
            allowed_edges = [(q, q) for q in range(n)]
        for (src_state, dst_state) in allowed_edges:
            mask[src_state, dst_state] = 0.0
        self.register_buffer("mask", mask)

    def _row_fallback(self, logits: torch.Tensor) -> torch.Tensor:
        _, n, _ = logits.shape
        row_has_finite = torch.isfinite(logits).any(dim=-1, keepdim=True)  # (B,n,1)
        if not row_has_finite.all():
            eye = torch.eye(n, device=logits.device).unsqueeze(0)          # (1,n,n)
            logits = torch.where(row_has_finite, logits, 10.0 * eye)       # force self-loop
        return logits

    def forward(self, predicate_truths_t: torch.Tensor,
                row_temperature: float = 0.5,
                literal_temperature: float = 0.8) -> torch.Tensor:
        batch_size = predicate_truths_t.size(0)
        guard_strengths = torch.stack(
            [edge(predicate_truths_t, literal_temperature) for edge in self.edges],
            dim=-1
        ).view(batch_size, self.n, self.n)  # (B,n,n)

        # Better gradients when guards are small: use logit(g) instead of log(g), clamp to (eps, 1-eps) for numerical safety
        # logits = torch.log(guard_strengths.clamp_min(1e-6)) + self.mask
        eps = 1e-6
        logits = torch.logit(guard_strengths.clamp(eps, 1.0 - eps)) + self.mask  # (B,n,n)

        logits = self._row_fallback(logits)
        transition_matrix_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return transition_matrix_t

    def forward_with_g(self, predicate_truths_t: torch.Tensor,
                       row_temperature: float = 0.5,
                       literal_temperature: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = predicate_truths_t.size(0)
        guard_strengths = torch.stack(
            [edge(predicate_truths_t, literal_temperature) for edge in self.edges],
            dim=-1
        ).view(batch_size, self.n, self.n)  # (B,n,n)

        # Better gradients when guards are small: use logit(g) instead of log(g), clamp to (eps, 1-eps) for numerical safety
        # logits = torch.log(guard_strengths.clamp_min(1e-6)) + self.mask
        eps = 1e-6
        logits = torch.logit(guard_strengths.clamp(eps, 1.0 - eps)) + self.mask  # (B,n,n)

        logits = self._row_fallback(logits)
        transition_matrix_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return transition_matrix_t, guard_strengths

    @torch.no_grad()
    def extract_all(self, names: List[str], ts: float = 0.75, ta: float = 0.2
                    ) -> Dict[Tuple[int, int], List[List[Tuple[str, bool]]]]:
        dnfs_by_edge: Dict[Tuple[int, int], List[List[Tuple[str, bool]]]] = {}
        for src_state in range(self.n):
            for dst_state in range(self.n):
                if not torch.isfinite(self.mask[src_state, dst_state]):
                    continue
                edge = self.edges[src_state * self.n + dst_state]
                dnfs_by_edge[(src_state + 1, dst_state + 1)] = edge.extract(names, ts, ta)
        return dnfs_by_edge

class NeSySFA(nn.Module):
    """
    DFA wrapper: transition layer + forward dynamic program (+ optional traces).
    """
    def __init__(self, n: int, m: int, R: int = 2,
                 allowed_edges: List[Tuple[int, int]] = None,
                 accept_state: int = 3):
        super().__init__()
        self.n = n
        self.num_states = n
        self.transition = TransitionLayer(n, m, R, allowed_edges)
        self.register_buffer("alpha0", F.one_hot(torch.tensor(0), num_classes=n).float().unsqueeze(0))
        self.register_buffer("accept", F.one_hot(torch.tensor(accept_state), num_classes=n).float())

    def forward_with_traces(self, predicate_truths_seq: torch.Tensor,
                            row_temperature: float = 0.5,
                            literal_temperature: float = 0.8):
        predicate_truths_seq = predicate_truths_seq.float()
        batch_size, seq_len, _ = predicate_truths_seq.shape

        alpha_t = self.alpha0.repeat(batch_size, 1)  # α_0
        state_beliefs = [alpha_t]
        transition_matrices = []

        for t in range(seq_len):
            M_t = self.transition(predicate_truths_seq[:, t, :],
                                  row_temperature=row_temperature,
                                  literal_temperature=literal_temperature)
            transition_matrices.append(M_t)
            alpha_t = torch.bmm(alpha_t.unsqueeze(1), M_t).squeeze(1)  # α_{t+1}
            state_beliefs.append(alpha_t)

        y_pred = (alpha_t * self.accept).sum(-1)  # ⟨α_T, A⟩
        return y_pred, state_beliefs[:-1], transition_matrices

    def forward_with_traces_and_g(self, predicate_truths_seq: torch.Tensor,
                                  row_temperature: float = 0.5,
                                  literal_temperature: float = 0.8):
        predicate_truths_seq = predicate_truths_seq.float()
        batch_size, seq_len, _ = predicate_truths_seq.shape

        alpha_t = self.alpha0.repeat(batch_size, 1)
        state_beliefs = [alpha_t]
        transition_matrices = []
        guard_strengths = []

        for t in range(seq_len):
            M_t, g_t = self.transition.forward_with_g(predicate_truths_seq[:, t, :],
                                                      row_temperature=row_temperature,
                                                      literal_temperature=literal_temperature)
            transition_matrices.append(M_t)
            guard_strengths.append(g_t)
            alpha_t = torch.bmm(alpha_t.unsqueeze(1), M_t).squeeze(1)
            state_beliefs.append(alpha_t)

        y_pred = (alpha_t * self.accept).sum(-1)
        return y_pred, state_beliefs[:-1], transition_matrices, guard_strengths

    def forward(self, predicate_truths_seq: torch.Tensor,
                row_temperature: float = 0.5,
                literal_temperature: float = 0.8) -> torch.Tensor:
        y_pred, _, _ = self.forward_with_traces(predicate_truths_seq,
                                                row_temperature=row_temperature,
                                                literal_temperature=literal_temperature)
        return y_pred

# --------------------------------------------------------------------------
# Helpers for training losses (unchanged)
# --------------------------------------------------------------------------
def compute_intended_guards_from_u(predicate_truths_seq: torch.Tensor
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optional weak priors ("event sketches") computed directly from predicates.

    NOTE (multivariate): applies to the **first dimension's** predicates if D>1.
    For MNIST digits we use (on dim 1):
      G1 ≈ even ∧ ge_7,     G2 ≈ odd ∧ le_5,     G3 ≈ le_2
    """
    m = predicate_truths_seq.shape[-1]
    if m < 22:
        raise ValueError("compute_intended_guards_from_u expects at least 22 predicates.")
    # First dimension slice
    even = predicate_truths_seq[:, :, 0]
    odd  = predicate_truths_seq[:, :, 1]
    le5  = predicate_truths_seq[:, :, 2 + 5]
    le2  = predicate_truths_seq[:, :, 2 + 2]
    ge7  = predicate_truths_seq[:, :, 2 + 10 + 7]
    G1 = even * ge7
    G2 = odd  * le5
    G3 = le2
    return G1, G2, G3

def row_entropy(transition_matrix: torch.Tensor) -> torch.Tensor:
    p = transition_matrix.clamp_min(1e-8)
    return -(p * p.log()).sum(-1).mean()

def pretty_clause(clause: List[Tuple[str, bool]]) -> str:
    return " ∧ ".join([name if positive else f"NOT({name})"
                       for name, positive in clause]) if clause else "TRUE"

def selector_sparsity_penalty(model) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=device); count = 0
    for edge in model.transition.edges:
        if hasattr(edge, "sel"):
            selector_probs = F.softmax(edge.sel, dim=-1)  # (R,m,3)
            penalty = penalty + (1.0 - selector_probs[..., 0]).mean()   # literals not ignored
            if hasattr(edge, "alog"):
                penalty = penalty + torch.sigmoid(edge.alog).mean()
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return penalty / count

def rule_constraint_penalty(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft penalties that discourage unsatisfiable / redundant clauses.
    Applies to DNF; skipped for pix/hybrid.
    """
    p_contra = torch.tensor(0.0, device=device)
    p_am1    = torch.tensor(0.0, device=device)
    cnt_c = 0; cnt_m = 0

    # indices per dimension chunk will be handled by *summing across all dims*
    # (we use only the positive channel; see notes in earlier versions)
    for edge in model.transition.edges:
        if not hasattr(edge, "sel"):   # skip pix/hybrid edges
            continue
        sel_probs = F.softmax(edge.sel, dim=-1)      # (R,m,K)
        pos_ch = sel_probs[..., 1]                   # (R,m)

        m = pos_ch.shape[-1]
        if m % 22 != 0:
            continue
        D = m // 22
        # build masks once per edge
        for d in range(D):
            base = d * 22
            idx_even, idx_odd = base + 0, base + 1
            idx_le = list(range(base + 2, base + 12))      # le_0..le_9
            idx_ge = list(range(base + 12, base + 22))     # ge_0..ge_9

            pos_even = pos_ch[:, idx_even]               # (R,)
            pos_odd  = pos_ch[:, idx_odd]                # (R,)
            p_contra += (pos_even * pos_odd).mean()

            le_pos = pos_ch[:, idx_le]                   # (R,10)
            ge_pos = pos_ch[:, idx_ge]                   # (R,10)
            mask_m_gt_k = torch.triu(torch.ones(10, 10, device=device), diagonal=1)
            pair = le_pos.unsqueeze(-1) * ge_pos.unsqueeze(-2)
            p_contra += (pair * mask_m_gt_k).mean()
            cnt_c += 1

            excess_le = F.relu(le_pos.sum(dim=-1) - 1.0).mean()
            excess_ge = F.relu(ge_pos.sum(dim=-1) - 1.0).mean()
            p_am1 += (excess_le + excess_ge) * 0.5
            cnt_m += 1

    if cnt_c > 0: p_contra = p_contra / cnt_c
    if cnt_m > 0: p_am1    = p_am1 / cnt_m
    return p_contra, p_am1

# --------------------------------------------------------------------------
# Evaluation helpers (F1, threshold sweep, edge usage)
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_f1_from_loader(model: NeSySFA, bank: PredicateBase, loader: DataLoader,
                           tau_row: float = 0.3, tau_gs: float = 0.7, threshold: float = 0.5):
    """
    Compute precision/recall/F1 for a fixed decision threshold.
    """
    model.eval()
    TP = FP = TN = FN = 0
    for batch in loader:
        xb, yb, _dim_names = (batch_ts_to_xy(batch))
        xb = xb.to(device); yb = yb.to(device)
        u_seq = u_from_probs_any(bank, xb)
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
def sweep_f1(model: NeSySFA, bank: PredicateBase, loader: DataLoader,
             tau_row: float = 0.3, tau_gs: float = 0.7, steps: int = 101):
    """
    Sweep the decision threshold and report the best-F1 point (and its P/R).
    """
    scores, labels = [], []
    for batch in loader:
        xb, yb, _ = (batch_ts_to_xy(batch))
        xb = xb.to(device); yb = yb.to(device)
        u_seq = u_from_probs_any(bank, xb)
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
def summarize_edge_usage(model: NeSySFA, bank: PredicateBase, loader: DataLoader,
                         tau_row: float = 0.4, tau_gs: float = 0.8, device: str = "cuda"):
    """
    Aggregate expected flow per edge to see which transitions are actually used.
    """
    n = model.transition.n
    usage = torch.zeros(n, n, device=device)
    total_sequences = 0
    model.eval()
    for batch in loader:
        xb, yb, _ = (batch_ts_to_xy(batch))
        xb = xb.to(device)
        u_seq = u_from_probs_any(bank, xb)
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

import torch

def combine_literals(lit_p: torch.Tensor,
                     combiner: str = "prod",
                     mask: torch.Tensor | None = None,
                     eps: float = 1e-6) -> torch.Tensor:
    """
    Combine literal probabilities within a clause.

    Args:
        lit_p: tensor of literal probabilities in [0,1], shape (..., L)
               (already includes negations, i.e., you've turned ¬x into (1 - p_x) upstream).
        combiner: 'prod' or 'logit-sum'
        mask: optional boolean/binary tensor same shape as lit_p indicating which literals are active.
              Inactive positions are treated as neutral elements (1.0 for product, 0.0 logit for logit-sum).
        eps: numerical clamp.

    Returns:
        clause probability tensor of shape (...,)
    """
    # Clamp away from 0/1 to avoid NaNs/inf in logit/product
    lit_p = lit_p.clamp(eps, 1.0 - eps)

    if mask is not None:
        # Ensure boolean
        if mask.dtype != torch.bool:
            mask = mask != 0

    if combiner == "prod":
        if mask is not None:
            # neutral element for product is 1
            lit_p = torch.where(mask, lit_p, torch.ones_like(lit_p))
        return lit_p.prod(dim=-1)

    elif combiner == "logit-sum":
        logits = torch.logit(lit_p)  # safe due to clamp
        if mask is not None:
            # neutral for sum-of-logits is 0
            logits = torch.where(mask, logits, torch.zeros_like(logits))
        return torch.sigmoid(logits.sum(dim=-1))

    else:
        raise ValueError(f"Unknown combiner: {combiner}")





#-----------------------------------------------------------------------------------------------------------------
#----------------------------------------- Debugging Diagnostics -------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def print_initial_row_stats(model, bank, loader, row_temp=0.8, lit_temp=1.0, n_batches=1):
    """
    Print mean transition probabilities at t=0 across the first few batches.
    Checks if rows are biased to self vs forward at initialization.
    """
    model.eval()
    it = iter(loader)
    for _ in range(n_batches):
        xb, yb, _ = batch_ts_to_xy(next(it))
        xb = xb.to(next(model.parameters()).device)
        u  = u_from_probs_any(bank, xb)
        _, _, Ms, _ = model.forward_with_traces_and_g(u, row_temperature=row_temp, literal_temperature=lit_temp)
        M0 = Ms[0].mean(0)  # (n,n)
        print("[row stats @t=0, mean over batch]")
        for q in range(model.n):
            row = "  ".join(f"P({q+1}->{j+1})={M0[q,j].item():.3f}" for j in range(model.n))
            print(f"  row {q+1}: {row}")

@torch.no_grad()
def print_EA_timeH(model, bank, loader, row_temp=0.6, lit_temp=0.9):
    """
    Print Expected Advances (EA) and time-entropy: are we progressing along the chain,
    and how concentrated is that progress over time?
    """
    model.eval()
    xb, yb, _ = batch_ts_to_xy(next(iter(loader)))
    xb = xb.to(next(model.parameters()).device)
    u  = u_from_probs_any(bank, xb)
    y, alphas, Ms, _ = model.forward_with_traces_and_g(u, row_temperature=row_temp, literal_temperature=lit_temp)

    # sum expected forward moves q->q+1
    adv_by_time = []
    for t, M_t in enumerate(Ms):
        αt = alphas[t]  # (B,n)
        a_t = torch.zeros(xb.size(0), device=xb.device)
        for q in range(model.n - 1):
            a_t += αt[:, q] * M_t[:, q, q+1]
        adv_by_time.append(a_t)
    adv_by_time = torch.stack(adv_by_time, 0)  # (T,B)

    EA = adv_by_time.sum(0).mean().item()
    p_time = adv_by_time / (adv_by_time.sum(0, keepdim=True) + 1e-6)
    timeH = (-(p_time.clamp_min(1e-8).log() * p_time).sum(0)).mean().item()
    print(f"[diag] EA≈{EA:.2f}  timeH≈{timeH:.2f}  ŷ_mean={y.mean().item():.3f}")

@torch.no_grad()
def print_guard_snapshot(model, bank, loader, row_temp=0.6, lit_temp=0.9, t_step=0):
    """
    Print mean guard strengths g(q->q') at a given time step.
    Useful to see if guards are all flat (≈0.5) or sharpening.
    """
    model.eval()
    xb, yb, _ = batch_ts_to_xy(next(iter(loader)))
    xb = xb.to(next(model.parameters()).device)
    u  = u_from_probs_any(bank, xb)
    _, _, _, Gs = model.forward_with_traces_and_g(u, row_temperature=row_temp, literal_temperature=lit_temp)
    G = Gs[t_step].mean(0)  # (n,n)
    print(f"[guards @t={t_step}] mean over batch")
    for q in range(model.n):
        row = "  ".join(f"{G[q,j].item():.3f}" for j in range(model.n))
        print(f"  row {q+1}: {row}")

@torch.no_grad()
def print_selector_stats(model):
    """
    Print average selector probabilities over {IGNORE, +, −} across all edges/clauses.
    If IGN≈1.0 early, sparsity is too strong or literal temperature is too low.
    """
    pos_means, neg_means, ign_means = [], [], []
    for e in getattr(model.transition, "edges", []):
        if not hasattr(e, "sel"):
            continue  # skip pix-only edges
        sel = F.softmax(e.sel, dim=-1)  # (R,m,3)
        ign_means.append(sel[...,0].mean().item())
        pos_means.append(sel[...,1].mean().item())
        neg_means.append(sel[...,2].mean().item())
    if pos_means:
        print(f"[selector] mean IGN={np.mean(ign_means):.3f}  POS={np.mean(pos_means):.3f}  NEG={np.mean(neg_means):.3f}")
#-----------------------------------------------------------------------------------------------------------------
#----------------------------------------- Debugging Diagnostics -------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------










# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _validate_allowed(allowed: List[Tuple[int, int]], num_states: int):
    out_deg = [0] * num_states
    for q, qp in allowed:
        out_deg[q] += 1
    bad = [q for q, d in enumerate(out_deg) if d == 0]
    if bad:
        raise ValueError(f"States with no outgoing edges: {bad}. Add at least a self-loop (q,q).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train δSFA with predicate guards.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument("--num-states", type=int, default=4, help="Number of states in the SFA.")
    parser.add_argument("--num-ors", type=int, default=1, help="Number of disjunctions in a guard DNF.")

    # Regularizer strengths and temperatures
    parser.add_argument("--lambda-flow", type=float, default=5e-3, help="Flow concentration weight.")
    parser.add_argument("--lambda-part", type=float, default=1e-2, help="Partition loss weight.")
    parser.add_argument("--part-xor",   type=float, default=1.0,  help="Mutual exclusion strength inside partition loss.")
    parser.add_argument("--lambda-prog",type=float, default=5e-2, help="Progress alignment weight (0 disables).")
    parser.add_argument("--tau-row",    type=float, default=1.0,  help="Training row-softmax temperature.")  # 0.4

    parser.add_argument("--tau-gs",     type=float, default=1.0,  help="Training DNF literal temperature.")

    parser.add_argument("--epochs",     type=int,   default=50,   help="Number of training epochs.")
    parser.add_argument("--train-path", type=str, required=True,  help="Path to training .pt (TensorSequence) file.")
    parser.add_argument("--test-path",  type=str, required=True,  help="Path to test .pt (TensorSequence) file.")
    parser.add_argument("--batch-size", type=int, default=64,     help="Batch size.")
    parser.add_argument("--extract-th", type=float, default=0.20, help="Literal threshold ts for extraction.")
    parser.add_argument("--extract-ta", type=float, default=0.20, help="Clause α threshold ta for extraction (DNF/hybrid).")
    parser.add_argument("--guard", choices=["dnf", "pix", "pix_hybrid"], default="dnf",
                        help="'dnf' = ternary-literal DNF; 'pix' = pix AND+OR; 'pix_hybrid' = pix AND + noisy-OR.")

    # Row-temperature annealing (optional)
    parser.add_argument("--tau-row-start", type=float, default=1.0, help="Initial row-softmax temperature (warmer = softer).")
    parser.add_argument("--tau-row-final", type=float, default=0.1, help="Final row-softmax temperature for the last epoch.")
    parser.add_argument("--tau-sched", choices=["linear", "cosine"], default="linear", help="Annealing schedule for tau_row over epochs.")

    # Any-time acceptance auxiliary loss (optional)
    parser.add_argument("--lambda-any", type=float, default=5e-3, help="Auxiliary weight for any-time acceptance BCE (0 disables).")

    parser.add_argument("--lambda-contradict", type=float, default=1e-3,
                        help="Weight for literal-level contradiction penalty (even∧odd, le_k∧ge_m with m>k). Set to 0 to disable.")
    parser.add_argument("--lambda-atmost-one", type=float, default=1e-3,
                        help="Weight for at-most-one penalty within threshold families (le_*, ge_*). Set to zero to disable.")

    parser.add_argument(
        "--keep-preds",
        type=str,
        default="",
        help="Comma-separated predicate names to KEEP (others dropped). "
             "Names can be 'd1_even' OR 'even_d1'. Leave empty to keep all."
    )
    parser.add_argument(
        "--drop-preds",
        type=str,
        default="",
        help="Comma-separated predicate names to DROP (others kept). "
             "Names can be 'd1_even' OR 'even_d1'. Ignored if --keep-preds is used."
    )

    # --- argparse additions ---
    parser.add_argument(
        "--lit-combiner",
        choices=["prod", "logit-sum"],
        default="prod",
        help="How to combine literal probabilities within a clause. "
             "'prod' = plain product; 'logit-sum' = sigmoid(sum(logit(p)))."
    )

    args = parser.parse_args()

    # Topology: by default a chain with self-loops + absorbing accept.
    allowed = utils.build_transitions(args.num_states, allow_backward=False, allow_jumps=False)
    _validate_allowed(allowed, num_states=args.num_states)

    # Load external TensorSequence datasets
    assert HAVE_EXT, "data_structs.get_data/get_data_loader not importable."
    train_data, test_data = get_data(args.train_path, args.test_path)
    train_loader = get_data_loader(train_data, args.batch_size, train=True)
    test_loader  = get_data_loader(test_data, args.batch_size,  train=False)
    print(f"[INFO] Loaded external TensorSequence data: {len(train_data)} train / {len(test_data)} test")

    # ---- Infer dimensionality D (and human-friendly names) from a single batch ----
    sample_batch = next(iter(train_loader))
    xb_sample, yb_sample, dim_names = batch_ts_to_xy(sample_batch)   # xb_sample: (B,T,D,10)
    D = xb_sample.shape[2]
    base_predicates = PredicateBase()

    #---------------------------------------------------------------------------------------------------------
    names = predicate_names_multi(dim_names, D)  # length = 22*D
    # --- Optional predicate gating: keep/drop by name ---
    keep_str = (args.keep_preds or "").strip()
    drop_str = (args.drop_preds or "").strip()

    if keep_str:
        keep_list = [s.strip() for s in keep_str.split(",") if s.strip()]
        idx_keep = _build_keep_indices(names, keep_list)
        ACTIVE_PRED_IDX = torch.tensor(idx_keep, dtype=torch.long)  # <- plain assignment
        names = [names[i] for i in idx_keep]
        print(f"[INFO] Keeping {len(idx_keep)} predicates: {', '.join(names)}")

    elif drop_str:
        drop_list = [s.strip() for s in drop_str.split(",") if s.strip()]
        idx_keep = _build_drop_indices(names, drop_list)
        ACTIVE_PRED_IDX = torch.tensor(idx_keep, dtype=torch.long)  # <- plain assignment
        names = [names[i] for i in idx_keep]
        print(f"[INFO] Dropping {len(drop_list)} → keeping {len(idx_keep)} predicates: {', '.join(names)}")

    else:
        ACTIVE_PRED_IDX = None
        print(f"[INFO] Using all predicates ({len(names)})")

    m_predicates = len(names)

    print(f"[INFO] Active predicates = {m_predicates}: {', '.join(names)}")

    # ---------------------------------------------------------------------------------------------------------





    # Build model (DNF by default; pix modes require δSFA_neuralDNF.py)
    if args.guard == "dnf":
        model = NeSySFA(n=args.num_states,
                        m=m_predicates,
                        R=args.num_ors,
                        allowed_edges=allowed,
                        accept_state=args.num_states - 1).to(device)
    else:
        assert HAVE_PIX, "NeSyDFA_Pix not found — make sure δSFA_neuralDNF.py is importable."
        use_or_unit = (args.guard == "pix")
        model = NeSyDFA_Pix(
            n=args.num_states,
            m=m_predicates, R=args.num_ors,
            allowed_edges=allowed,
            accept_state=args.num_states - 1,
            use_or_unit=use_or_unit,
            slope_and=1.5, slope_or=1.5
        ).to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = utils.count_parameters(model, verbose=True)
    print(parameters, params)
    utils.count_by_child(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- one-shot pre-training diagnostics ---
    # print_initial_row_stats(model, base_predicates, train_loader, row_temp=0.8, lit_temp=1.0, n_batches=1)
    # print_EA_timeH(model, base_predicates, train_loader, row_temp=0.8, lit_temp=1.0)
    # print_guard_snapshot(model, base_predicates, train_loader, row_temp=0.8, lit_temp=1.0, t_step=0)
    # print_selector_stats(model)

    # ------------------------------------------------------------------
    # Regularizers (kept identical to your previous version)
    # ------------------------------------------------------------------
    lambda_ent   = 1e-2
    lambda_sel   = 0.01 # 1e-3  # 0.1
    if args.guard in ("pix", "pix_hybrid"):
        lambda_sel *= 0.05
    lambda_flow  = args.lambda_flow
    lambda_part  = args.lambda_part
    part_xor     = args.part_xor
    lambda_prog  = args.lambda_prog

    # Temperatures (train vs eval)
    row_temperature_train     = args.tau_row
    literal_temperature_train = args.tau_gs
    row_temperature_eval      = 0.3
    literal_temperature_eval  = 0.7

    # --------------------------
    # Evaluation loss (val BCE)
    # --------------------------
    def eval_loss() -> float:
        model.eval()
        total_loss, total_examples = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                xb, yb, _ = (batch_ts_to_xy(batch))
                xb = xb.to(device); yb = yb.to(device)
                predicate_truths_seq = u_from_probs_any(base_predicates, xb)
                y_pred = model(predicate_truths_seq,
                               row_temperature=row_temperature_eval,
                               literal_temperature=literal_temperature_eval)
                total_loss += bce_safe(y_pred, yb).item() * xb.size(0)
                total_examples += xb.size(0)
        return total_loss / max(1, total_examples)

    # --------------------------
    # Training loop
    # --------------------------
    train_hist, val_hist = [], []
    print(f"[INFO] D={D}  m=22*D={m_predicates}  dim_names={dim_names}")
    print(f"[INFO] λ_flow={lambda_flow:.2e}  λ_part={lambda_part:.2e}  part_xor={part_xor:.2f}  "
          f"λ_prog={lambda_prog:.2e}  τ_row={row_temperature_train:.2f}  τ_gs={literal_temperature_train:.2f}")

    def _anneal_tau_row(epoch_idx: int, num_epochs_minus1: int,
                        tau0: float, tau1: float, schedule: str = "linear") -> float:
        if schedule == "linear":
            s = epoch_idx / max(1, num_epochs_minus1)
            return tau0 + (tau1 - tau0) * s
        s = epoch_idx / max(1, num_epochs_minus1)
        return tau1 + 0.5 * (tau0 - tau1) * (1 + math.cos(math.pi * s))

    for epoch in range(1, args.epochs + 1):

        row_temperature_train = _anneal_tau_row(epoch - 1, args.epochs - 1,
                                                args.tau_row_start, args.tau_row_final, args.tau_sched)
        model.train()
        epoch_loss_sum, epoch_examples = 0.0, 0

        for batch in train_loader:
            xb, yb, _ = (batch_ts_to_xy(batch))      # (B,T,D,10), (B,)
            xb = xb.to(device); yb = yb.to(device)

            # (1) Map digit probabilities → predicate truths for the whole sequence
            predicate_truths_seq = u_from_probs_any(base_predicates, xb)  # (B,T,22*D)

            if epoch == 1 and epoch_examples == 0:  # print only on first batch of epoch 1
                print(f"[DEBUG] u_seq shape = {tuple(predicate_truths_seq.shape)}  "
                      f"(last dim should be {len(names)})")

            # (2) Forward DP with traces for auxiliary losses
            y_pred, state_beliefs, transition_matrices, guard_strengths = \
                model.forward_with_traces_and_g(predicate_truths_seq,
                                                row_temperature=row_temperature_train,
                                                literal_temperature=literal_temperature_train)

            # (3) Primary loss
            loss_bce = bce_safe(y_pred, yb)

            # (5) Selector sparsity
            loss_sel = selector_sparsity_penalty(model)

            # (8) Row partition: self vs forward mutual exclusivity in raw g
            part_terms = []
            for t, g_raw in enumerate(guard_strengths):
                for q in range(model.n - 1):
                    if torch.isfinite(model.transition.mask[q, q]) and torch.isfinite(model.transition.mask[q, q+1]):
                        g_self = g_raw[:, q, q]
                        g_fwd  = g_raw[:, q, q+1]
                        part_terms.append(((g_self + g_fwd - 1.0)**2 + part_xor * (g_self * g_fwd)).mean())
            loss_part = torch.stack(part_terms).mean() if part_terms else torch.tensor(0.0, device=device)

            # loss = (loss_bce + lambda_sel * loss_sel + lambda_part * loss_part)
            loss = (loss_bce + lambda_sel * loss_sel )
            # loss = (loss_bce)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * xb.size(0)
            epoch_examples += xb.size(0)

        train_hist.append(epoch_loss_sum / epoch_examples)
        val_loss = eval_loss()
        val_hist.append(val_loss)
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_hist[-1]:.4f}  val={val_loss:.4f}")

        # end of each epoch (after printing train/val)
        """
        if epoch in (1, 2, 5, 10) or (epoch % 10 == 0):
            print_selector_stats(model)
            # use *current* training temps to see what the DP sees now
            print_EA_timeH(model, base_predicates, train_loader, row_temp=row_temperature_train,
                           lit_temp=literal_temperature_train)
            print_guard_snapshot(model, base_predicates, train_loader, row_temp=row_temperature_train,
                                 lit_temp=literal_temperature_train, t_step=0)
        """

    # --------------------------
    # Curves
    # --------------------------
    plt.figure()
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="test")
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.title("Loss Curves")
    plt.tight_layout(); plt.savefig("loss_curve.png")

    # Edge usage summary on train set
    summarize_edge_usage(model, base_predicates, train_loader)

    # ---------------------------------------------
    # Extraction thresholds explanation:
    #
    #   ts  (literal threshold)
    #       • DNF guards: threshold on the literal *selector* softmax per predicate.
    #         A literal is included in a clause only if max{P(+),P(−)} ≥ ts.
    #         (Selector is over {IGNORE,+,−}; polarity = argmax over {+,−}.)
    #
    #   ta  (clause activation threshold)
    #       • DNF guards: threshold on the clause strength α_r = σ(a_r). Keep clause if α_r ≥ ta.
    # ---------------------------------------------
    # Use the filtered names that match the active predicate channels
    dnfs_by_edge = model.transition.extract_all(
        names,  # <-- this is the filtered list built after --keep-preds / --drop-preds
        ts=args.extract_th,
        ta=args.extract_ta
    )


    print("\nLearned guards (DNFs) per ALLOWED edge (q->q'):\n")
    for (q, qp), clauses in dnfs_by_edge.items():
        if not clauses:
            continue
        print(f"{q}->{qp}:  " + "  OR  ".join(pretty_clause(c) for c in clauses))

    # Optional graph viz
    if HAVE_NX:
        G = nx.DiGraph()
        for s in range(1, args.num_states + 1):
            G.add_node(s, label=f"{s}" + (" (accept)" if s == args.num_states else ""))
        for (q, qp), clauses in dnfs_by_edge.items():
            if not clauses:
                continue
            G.add_edge(q, qp, label=" OR ".join(pretty_clause(c) for c in clauses))
        pos = nx.spring_layout(G, seed=2)
        plt.figure()
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_size=8)
        plt.title("Extracted SFA"); plt.axis('off'); plt.tight_layout()
        plt.savefig("extracted_sfa.png")

    print("\nSaved:")
    print(" - loss_curve.png")
    if HAVE_NX:
        print(" - extracted_sfa.png")

    # --------------------------
    # Evaluation on test set
    # --------------------------
    test_loader = get_data_loader(test_data, batch_size=512, train=False)

    # 1) Standard F1 at threshold=0.5
    compute_f1_from_loader(
        model=model,
        bank=base_predicates,
        loader=test_loader,
        tau_row=0.3,
        tau_gs=0.7,
        threshold=0.5
    )

    # 2) Best-F1 threshold sweep
    sweep_f1(model, base_predicates, test_loader, tau_row=0.3, tau_gs=0.7, steps=101)
