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
• DNFEdge            : differentiable DNF guard per edge (ternary literals)
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
# even, odd, le_0..le_9, ge_0..ge_9  (m = 22 predicates)
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

def predicate_names() -> List[str]:
    """Human-readable names for the 22 base predicates in `PredicateBase`."""
    return ["even","odd"] + [f"le_{k}" for k in range(10)] + [f"ge_{k}" for k in range(10)]

def u_from_probs(bank: PredicateBase, prob_seq: torch.Tensor) -> torch.Tensor:
    """
    Convert a sequence of digit probabilities into per-time predicate truths.

    Args
    ----
    bank : PredicateBase
        Predicate mapper.
    prob_seq : (batch_size, seq_len, 10) float32
        Digit probabilities for each time step.

    Returns
    -------
    predicate_truths_seq : (batch_size, seq_len, 22) float32
        Per-time predicate soft-truths for the sequence.
    """
    batch_size, seq_len, _ = prob_seq.size()
    return torch.stack([bank.compute(prob_seq[:, t, :]) for t in range(seq_len)], dim=1)

# --------------------------------------------------------------------------
# TensorSequence utilities: read symbolic digits & labels and make (B,T,10)
# --------------------------------------------------------------------------
def _extract_digits_from_ts(ts: "TensorSequence") -> Tuple[List[int], int]:
    """
    Read the symbolic digit sequence and label from a TensorSequence (univariate only).

    Args
    ----
    ts : TensorSequence
        An object holding the sequence of image labels and a sequence label.

    Returns
    -------
    digits : List[int]
        The per-time symbolic digits.
    y      : int
        The sequence label in {0,1}.
    """
    seq_len = ts.seq_length
    assert ts.dimensionality == 1, "Current SFA expects 1-D sequences (dimensionality==1)."
    digits = []
    for t in range(seq_len):
        # image_labels[t][0] is a dict like {'d1': <digit_int>}
        digit_val = next(iter(ts.image_labels[t][0].values()))
        digits.append(int(digit_val))
    y = int(ts.seq_label)
    return digits, y

def _digits_to_probs(digits: List[int], sharp: float = 0.98) -> np.ndarray:
    """Vectorize a whole sequence of symbolic digits into softmax-like probs."""
    return np.stack([cnn_probs(d, sharp=sharp) for d in digits], axis=0).astype(np.float32)

def batch_ts_to_xy(batch: List["TensorSequence"], sharp: float = 0.98) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a list of TensorSequence to tensors usable by the model.

    Returns
    -------
    xb : (batch_size, seq_len, 10) float32
        Per-time digit probabilities.
    yb : (batch_size,) float32
        Sequence labels in {0,1}.
    """
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
    """
    Stable binary cross-entropy that clamps NaN/Inf and bounds probabilities.

    Args
    ----
    y_pred : (batch_size,) float
        Predicted acceptance probability in [0,1], may contain NaN/Inf.
    y_true : (batch_size,) float
        Target label ∈ {0,1}.

    Returns
    -------
    loss : scalar tensor
        BCE(y_pred, y_true) after clamping y_pred∈(1e-6,1-1e-6), y_true∈[0,1].

    Why
    ---
    NaN/Inf can appear when a softmax row becomes degenerate or intermediate
    logs overflow. Clamping avoids device-side asserts in CUDA kernels.
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

    Structure
    ---------
    • Literals are ternary per predicate: {IGNORE, +, −}.
      For predicate truth u_j∈[0,1], a literal evaluates to:
          L_j = s_ignore + s_pos * u_j + s_neg * (1 − u_j),
      where s_* are selector probabilities (softmax over {IGNORE,+,−}).

    • Clause r is a product of its literals:
          C_r = ∏_{j=1..m} L_{rj}.

    • Clause strength α_r = σ(a_r) (learned per clause).
      Guard activation (noisy-OR over clauses):
          g(x_t) = 1 − ∏_r [ 1 − α_r · C_r ]    ∈ [0,1].

    Parameters
    ----------
    m : int
        Number of base predicates (22 for MNIST-e/o/≤/≥).
    R : int
        Maximum number of clauses in the DNF for this edge.

    Learnable tensors
    -----------------
    sel  : (R, m, 3)  — raw logits → selector softmax over {IGNORE, +, −}
    alog : (R,)       — raw logits → α via sigmoid
    """
    def __init__(self, m: int, R: int = 2):
        super().__init__()
        self.m = m
        self.R = R
        self.sel  = nn.Parameter(torch.zeros(R, m, 3))
        self.alog = nn.Parameter(torch.zeros(R))
        with torch.no_grad():
            self.sel[..., 0] = 1.0   # bias selectors toward IGNORE initially
            self.alog[:]      = -1.0 # weak clauses initially (α ≈ 0.27)

    def forward(self, predicate_truths_t: torch.Tensor, literal_temperature: float = 0.8) -> torch.Tensor:
        """
        Evaluate the guard g(x_t) for a batch of predicate truth vectors at time t.

        Args
        ----
        predicate_truths_t : (batch_size, m) float
            Base predicate values for this time step.
        literal_temperature : float
            Temperature for the selector softmax over {IGNORE,+,−}.
            Lower → sharper literal choices; higher → softer.

        Returns
        -------
        g_t : (batch_size,) float
            Guard activation probability for this edge at time t.

        Math
        ----
        See class docstring for the exact formulas of literals, clauses, and g(x_t).
        """
        u_t = predicate_truths_t.float()                                # (B,m)
        selector = F.softmax(self.sel / max(literal_temperature, 1e-6), dim=-1)  # (R,m,3)

        # Build literals: L = s_ignore + s_pos*u + s_neg*(1-u)
        u = u_t.unsqueeze(1)                                            # (B,1,m)
        literal_vals = (selector[..., 0].unsqueeze(0)
                        + selector[..., 1].unsqueeze(0) * u
                        + selector[..., 2].unsqueeze(0) * (1.0 - u))    # (B,R,m)

        clause_vals = torch.prod(literal_vals, dim=-1)                  # (B,R)
        alpha = torch.sigmoid(self.alog).unsqueeze(0)                   # (1,R)
        g_t = 1.0 - torch.prod(1.0 - alpha * clause_vals, dim=-1)       # (B,)
        return g_t.float()

    @torch.no_grad()
    def extract(self, names: List[str], ts: float = 0.75, ta: float = 0.2) -> List[List[Tuple[str, bool]]]:
        """
        Extract a symbolic DNF (list of clauses; each clause = list of (name, polarity)).

        Args
        ----
        names : List[str]
            Predicate names (length m).
        ts : float
            Literal selector threshold. Literal j is included if:
                max{P(+), P(−)} ≥ ts, and polarity is argmax over {+,−}.
        ta : float
            Clause activation threshold on α_r = σ(a_r). Clause is kept if α_r ≥ ta.

        Returns
        -------
        clauses : List[List[(name, polarity)]]
            The extracted DNF (empty if nothing survives thresholds).
        """
        selector_probs = F.softmax(self.sel, dim=-1).cpu().numpy()  # (R,m,3)
        alpha = torch.sigmoid(self.alog).cpu().numpy()              # (R,)
        dnfs: List[List[Tuple[str, bool]]] = []
        for r in range(self.R):
            if alpha[r] < ta:
                continue
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

    For each time t and source state q, we compute logits for all destination states q':
        logits[q,q'] = log g_{q→q'}(x_t) + mask[q,q']
    then apply a **row softmax** with temperature τ_row to get a transition distribution
    over q' from q. This yields M_t ∈ ℝ^{n×n} with rows summing to 1.

    Safety: if an entire row would be −inf (no allowed edges), we *fallback* by forcing a
    strong self-loop on that row so softmax remains well-defined.

    Parameters
    ----------
    n : int
        Number of automaton states.
    m : int
        Number of base predicates.
    R : int
        Max number of clauses per DNF guard (per edge).
    allowed_edges : List[(int,int)]
        List of (src_state, dst_state) pairs that are structurally allowed.
        All others are masked to −inf in the logits.
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int, int]] = None):
        super().__init__()
        self.n = n                              # keep for compatibility elsewhere
        self.num_states = n
        self.num_predicates = m
        self.max_clauses = R

        # One DNFEdge per directed (q→q') pair
        self.edges = nn.ModuleList([DNFEdge(m, R) for _ in range(n * n)])

        # Mask: 0 for allowed edges, −inf for disallowed (added to logits before softmax)
        mask = torch.full((n, n), float('-inf'))
        if allowed_edges is None:
            allowed_edges = [(q, q) for q in range(n)]  # default: only self-loops
        for (src_state, dst_state) in allowed_edges:
            mask[src_state, dst_state] = 0.0
        self.register_buffer("mask", mask)

    def _row_fallback(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Ensure no softmax row is degenerate.

        If a row has all entries −inf (e.g., all disallowed or guards ≈ 0), replace that
        row with a large logit on the diagonal (self-loop), which guarantees a valid softmax.

        Args
        ----
        logits : (batch_size, n, n) float

        Returns
        -------
        logits_fixed : (batch_size, n, n) float
        """
        _, n, _ = logits.shape
        row_has_finite = torch.isfinite(logits).any(dim=-1, keepdim=True)  # (B,n,1)
        if not row_has_finite.all():
            eye = torch.eye(n, device=logits.device).unsqueeze(0)          # (1,n,n)
            logits = torch.where(row_has_finite, logits, 10.0 * eye)       # force self-loop
        return logits

    def forward(self, predicate_truths_t: torch.Tensor,
                row_temperature: float = 0.5,
                literal_temperature: float = 0.8) -> torch.Tensor:
        """
        Compute the per-time transition matrix M_t from predicate truths.

        Args
        ----
        predicate_truths_t : (batch_size, m) float
            Base predicate truths for this time step.
        row_temperature : float
            Temperature for the row softmax over destinations q' (sharper when small).
        literal_temperature : float
            Temperature for the literal selectors inside each DNF guard.

        Returns
        -------
        transition_matrix_t : (batch_size, n, n) float
            Row-stochastic transition matrix for this time step.
        """
        batch_size = predicate_truths_t.size(0)
        # Evaluate all guards, one per edge (q→q')
        guard_strengths = torch.stack(
            [edge(predicate_truths_t, literal_temperature) for edge in self.edges],
            dim=-1
        ).view(batch_size, self.n, self.n)  # (B,n,n)

        # Logits = log g + mask, then row-softmax with temperature
        logits = torch.log(guard_strengths.clamp_min(1e-6)) + self.mask
        logits = self._row_fallback(logits)
        transition_matrix_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return transition_matrix_t

    def forward_with_g(self, predicate_truths_t: torch.Tensor,
                       row_temperature: float = 0.5,
                       literal_temperature: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        As `forward`, but also return raw guard strengths g_{q→q'}(x_t).

        Returns
        -------
        transition_matrix_t : (batch_size, n, n) float
        guard_strengths_t   : (batch_size, n, n) float
        """
        batch_size = predicate_truths_t.size(0)
        guard_strengths = torch.stack(
            [edge(predicate_truths_t, literal_temperature) for edge in self.edges],
            dim=-1
        ).view(batch_size, self.n, self.n)  # (B,n,n)

        logits = torch.log(guard_strengths.clamp_min(1e-6)) + self.mask
        logits = self._row_fallback(logits)
        transition_matrix_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return transition_matrix_t, guard_strengths

    @torch.no_grad()
    def raw_gates(self, predicate_truths_t: torch.Tensor, literal_temperature: float = 0.8) -> torch.Tensor:
        """Utility to get g_{q→q'}(x_t) without softmax/logits for inspection."""
        batch_size = predicate_truths_t.size(0)
        guard_strengths = torch.stack(
            [edge(predicate_truths_t, literal_temperature) for edge in self.edges],
            dim=-1
        ).view(batch_size, self.n, self.n)
        return guard_strengths

    @torch.no_grad()
    def extract_all(self, names: List[str], ts: float = 0.75, ta: float = 0.2
                    ) -> Dict[Tuple[int, int], List[List[Tuple[str, bool]]]]:
        """
        Extract readable DNFs for all **allowed** edges.

        Args
        ----
        names : List[str]
            Predicate names.
        ts : float
            Literal selector threshold (see DNFEdge.extract).
        ta : float
            Clause α threshold (see DNFEdge.extract).

        Returns
        -------
        dnfs_by_edge : Dict[(q,q'), List[clause]]
            Mapping from 1-based edge indices to DNF clauses.
        """
        dnfs_by_edge: Dict[Tuple[int, int], List[List[Tuple[str, bool]]]] = {}
        for src_state in range(self.n):
            for dst_state in range(self.n):
                if not torch.isfinite(self.mask[src_state, dst_state]):
                    continue
                edge = self.edges[src_state * self.n + dst_state]
                dnfs_by_edge[(src_state + 1, dst_state + 1)] = edge.extract(names, ts, ta)
        return dnfs_by_edge

class NeSyDFA(nn.Module):
    """
    DFA wrapper: transition layer + forward dynamic program (+ optional traces).

    Notation
    --------
    • α_t ∈ Δ^n : belief (row vector) over states after consuming t symbols.
      α_0 is one-hot on the start state (state 0).
    • M_t ∈ ℝ^{n×n} : row-stochastic transition at time t.
    • Accept set A: one-hot vector for the accepting state.

    Recurrence
    ----------
        α_{t+1} = α_t · M_t
        ŷ       = ⟨α_T, A⟩

    Methods
    -------
    forward_with_traces(u_seq, τ_row, τ_gs)
        Returns final ŷ, per-step α_t (t=0..T−1), and M_t.
    forward_with_traces_and_g(u_seq, τ_row, τ_gs)
        As above, plus raw guard strengths g_t.
    forward(u_seq, τ_row, τ_gs)
        Only the final acceptance probability ŷ.
    """
    def __init__(self, n: int, m: int, R: int = 2,
                 allowed_edges: List[Tuple[int, int]] = None,
                 accept_state: int = 3):
        super().__init__()
        self.n = n
        self.num_states = n
        self.transition = TransitionLayer(n, m, R, allowed_edges)
        # α_0 : start at state 0 for every sequence in the batch
        self.register_buffer("alpha0", F.one_hot(torch.tensor(0), num_classes=n).float().unsqueeze(0))
        # A   : one-hot for the accepting state
        self.register_buffer("accept", F.one_hot(torch.tensor(accept_state), num_classes=n).float())

    def forward_with_traces(self, predicate_truths_seq: torch.Tensor,
                            row_temperature: float = 0.5,
                            literal_temperature: float = 0.8):
        """
        Run the forward DP and return traces (α_t and M_t).

        Args
        ----
        predicate_truths_seq : (batch_size, seq_len, m) float
            Predicate truths for each time step of each sequence.
        row_temperature : float
            Row-softmax temperature (TransitionLayer).
        literal_temperature : float
            DNF literal selector temperature (TransitionLayer/DNFEdge).

        Returns
        -------
        y_pred        : (batch_size,) float
            Final acceptance probability per sequence.
        state_beliefs : List[(batch_size, n)]
            α_t for t = 0..T−1 (excludes the terminal α_T to keep length T).
        transition_matrices : List[(batch_size, n, n)]
            M_t for t = 0..T−1.
        """
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
        """
        As `forward_with_traces`, but also return raw guard strengths per time.

        Returns
        -------
        y_pred        : (batch_size,)
        state_beliefs : List[(batch_size, n)]
        transition_matrices : List[(batch_size, n, n)]
        guard_strengths     : List[(batch_size, n, n)]
        """
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
        """
        Convenience wrapper returning only the final acceptance probability ŷ."""
        y_pred, _, _ = self.forward_with_traces(predicate_truths_seq,
                                                row_temperature=row_temperature,
                                                literal_temperature=literal_temperature)
        return y_pred

# --------------------------------------------------------------------------
# Helpers for training losses
# --------------------------------------------------------------------------
def compute_intended_guards_from_u(predicate_truths_seq: torch.Tensor
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optional weak priors ("event sketches") computed directly from predicates.

    For MNIST digits we use:
      G1 ≈ even ∧ ge_7,     G2 ≈ odd ∧ le_5,     G3 ≈ le_2

    These encourage advances 0→1, 1→2, 2→3 to happen where the corresponding
    event is likely. Disable their influence by setting --lambda-prog 0.

    Args
    ----
    predicate_truths_seq : (batch_size, seq_len, 22)

    Returns
    -------
    (G1, G2, G3) : each (batch_size, seq_len) in [0,1]
    """
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
    """
    Row-wise entropy H = −Σ_j p_j log p_j averaged over sequences and rows.

    Args
    ----
    transition_matrix : (batch_size, n, n)
        A row-stochastic matrix (rows sum to 1).

    Returns
    -------
    avg_entropy : scalar tensor
        Mean entropy across all rows in the batch.
    """
    p = transition_matrix.clamp_min(1e-8)
    return -(p * p.log()).sum(-1).mean()

def pretty_clause(clause: List[Tuple[str, bool]]) -> str:
    """Format a clause (list of literals) for printing."""
    return " ∧ ".join([name if positive else f"NOT({name})"
                       for name, positive in clause]) if clause else "TRUE"

def selector_sparsity_penalty(model) -> torch.Tensor:
    """
    Promote compact guards by penalizing *used* literals/clauses.

    DNF guards
    ----------
    For each edge, `sel[r,j,:]` is a softmax over {IGNORE, +, −}.
    We add:
        • literal-use penalty:  mean(1 − P(IGNORE)) across (r,j),
          which encourages assigning mass to IGNORE (i.e., *dropping* literals);
        • clause-use penalty:  mean(σ(a_r)) across r,
          which encourages smaller/ fewer active clauses.

    pix / pix_hybrid (if enabled)
    -----------------------------
    We add an L1 on AND-unit weights (|w|) and, if present, on α (σ(a_r)).
    This shrinks small contributions to zero and behaves like a sparsity prior.

    Returns
    -------
    penalty : scalar tensor
        Average penalty across all edges (normalized by count so its scale
        is relatively invariant to graph size).
    """
    penalty = torch.tensor(0.0, device=device)
    count = 0

    for edge in model.transition.edges:
        # DNF case
        if hasattr(edge, "sel"):
            selector_probs = F.softmax(edge.sel, dim=-1)  # (R,m,3)
            penalty = penalty + (1.0 - selector_probs[..., 0]).mean()   # literals not ignored
            if hasattr(edge, "alog"):                                   # active clauses
                penalty = penalty + torch.sigmoid(edge.alog).mean()
            count += 1

        # pix / hybrid case (if present)
        elif hasattr(edge, "and_units"):
            for and_unit in edge.and_units:
                penalty = penalty + and_unit.w.abs().mean()
                count += 1
            if hasattr(edge, "or_unit"):
                penalty = penalty + edge.or_unit.w.abs().mean()
                count += 1
            if hasattr(edge, "alog"):
                penalty = penalty + torch.sigmoid(edge.alog).mean()
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return penalty / count


def rule_constraint_penalty(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute two soft penalties that discourage unsatisfiable or redundant clauses.

    Returns
    -------
    (p_contradict, p_am1) : (scalar, scalar)
        p_contradict: penalizes even∧odd and le_k∧ge_m with m>k inside the same clause.
        p_am1:        penalizes selecting >1 literal from a threshold family (le_* or ge_*) per clause.
    Notes
    -----
    • Applies only to DNF guards (hasattr(edge, "sel")). Pix guards are skipped.
    • Uses selector softmax over {IGNORE, +, (− if enabled)}; we look only at the **positive** channel.
    """
    p_contra = torch.tensor(0.0, device=device)
    p_am1    = torch.tensor(0.0, device=device)
    cnt_c = 0
    cnt_m = 0

    # Predicate indices in your PredicateBase
    idx_even, idx_odd = 0, 1
    idx_le = list(range(2, 12))      # le_0..le_9
    idx_ge = list(range(12, 22))     # ge_0..ge_9

    # Mask for le_k ∧ ge_m with m>k (unsatisfiable): shape (10,10)
    # le index i ↔ k=i, ge index j ↔ m=j
    mask_m_gt_k = torch.triu(torch.ones(10, 10, device=device), diagonal=1)  # 1 if m>k else 0

    for edge in model.transition.edges:
        if not hasattr(edge, "sel"):   # skip pix/hybrid edges
            continue
        sel_probs = F.softmax(edge.sel, dim=-1)      # (R,m,K)
        pos_ch = sel_probs[..., 1]                   # (R,m) positive literal prob
        # ---- (1) contradictions ----
        pos_even = pos_ch[:, idx_even]               # (R,)
        pos_odd  = pos_ch[:, idx_odd]                # (R,)
        # even ∧ odd in same clause
        p_contra += (pos_even * pos_odd).mean()
        # le_k ∧ ge_m with m>k in same clause
        le_pos = pos_ch[:, idx_le]                   # (R,10)
        ge_pos = pos_ch[:, idx_ge]                   # (R,10)
        # pairwise product with mask m>k
        # (R,10,10): le_pos[...,k] * ge_pos[...,m]
        pair = le_pos.unsqueeze(-1) * ge_pos.unsqueeze(-2)
        p_contra += (pair * mask_m_gt_k).mean()
        cnt_c += 1

        # ---- (2) at-most-one per family ----
        # penalize excess over 1: relu(sum(pos) - 1)
        excess_le = F.relu(le_pos.sum(dim=-1) - 1.0).mean()  # (R,) → scalar
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
def compute_f1_from_loader(model: NeSyDFA, bank: PredicateBase, loader: DataLoader,
                           tau_row: float = 0.3, tau_gs: float = 0.7, threshold: float = 0.5):
    """
    Compute precision/recall/F1 for a fixed decision threshold.

    Args
    ----
    model : NeSyDFA
    bank  : PredicateBase
    loader: DataLoader yielding either (xb, yb) or [TensorSequence,...]
    tau_row : float
        Row-softmax temperature for evaluation.
    tau_gs  : float
        Literal temperature for evaluation.
    threshold : float
        Decision threshold on acceptance probability ŷ.

    Prints confusion matrix and returns (precision, recall, F1).
    """
    model.eval()
    TP = FP = TN = FN = 0
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
    """
    Sweep the decision threshold and report the best-F1 point (and its P/R).
    """
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

    For each time t we accumulate:
        flow_t(q→q') = E[ α_t(q) · M_t(q,q') ]  (expectation over the batch)
    and average across time and batches.

    Prints the top edges by usage and returns the (n,n) usage matrix.
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
    """
    Ensure that every state has at least one outgoing edge; otherwise softmax rows
    become undefined. Recommend adding (q,q) for all q and (accept, accept).
    """
    out_deg = [0] * num_states
    for q, qp in allowed:
        out_deg[q] += 1
    bad = [q for q, d in enumerate(out_deg) if d == 0]
    if bad:
        raise ValueError(f"States with no outgoing edges: {bad}. "
                         f"Add at least a self-loop (q,q).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeSyDFA with predicate guards (external data only).")
    # Regularizer strengths and temperatures
    parser.add_argument("--lambda-flow", type=float, default=5e-3, help="Flow concentration weight.")
    parser.add_argument("--lambda-part", type=float, default=1e-2, help="Partition loss weight.")
    parser.add_argument("--part-xor",   type=float, default=1.0,  help="Mutual exclusion strength inside partition loss.")
    parser.add_argument("--lambda-prog",type=float, default=5e-2, help="Progress alignment weight (0 disables).")
    parser.add_argument("--tau-row",    type=float, default=0.4,  help="Training row-softmax temperature.")
    parser.add_argument("--tau-gs",     type=float, default=0.8,  help="Training DNF literal temperature.")
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
    parser.add_argument("--tau-row-final", type=float, default=0.3, help="Final row-softmax temperature for the last epoch.")
    parser.add_argument("--tau-sched", choices=["linear", "cosine"], default="linear", help="Annealing schedule for tau_row over epochs.")

    # Any-time acceptance auxiliary loss (optional)
    parser.add_argument("--lambda-any", type=float, default=5e-3, help="Auxiliary weight for any-time acceptance BCE (0 disables).")

    parser.add_argument("--lambda-contradict", type=float, default=1e-3,
                        help="Weight for literal-level contradiction penalty (even∧odd, le_k∧ge_m with m>k). Set to 0 to disable.")
    parser.add_argument("--lambda-atmost-one", type=float, default=1e-3,
                        help="Weight for at-most-one penalty within threshold families (le_*, ge_*). Set to zero to disable.")
                        
    # PIX logical-bias annealing (controls hardness of AND/OR gates)
    parser.add_argument("--pix-bias-start", type=float, default=0.20,
                        help="Initial strength for pix logical bias (0..1). Lower = softer gates early.")
    parser.add_argument("--pix-bias-final", type=float, default=1.00,
                        help="Final strength for pix logical bias.")
    parser.add_argument("--pix-bias-sched", choices=["linear", "cosine"], default="linear",
                        help="Annealing schedule for pix logical bias over epochs.")                    

    args = parser.parse_args()

    # Topology: by default a chain with self-loops + absorbing accept.
    allowed = [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(3,3)]  #,   (2,1), (2,0), (1,0)     ,   (2,1), (2,0), (1,0), (0,3), (0,2), (1,3)
    _validate_allowed(allowed, num_states=4)

    # Load external TensorSequence datasets
    assert HAVE_EXT, "data_structs.get_data/get_data_loader not importable."
    train_data, test_data = get_data(args.train_path, args.test_path)
    train_loader = get_data_loader(train_data, args.batch_size, train=True)
    test_loader  = get_data_loader(test_data, args.batch_size,  train=False)
    print(f"[INFO] Loaded external TensorSequence data: {len(train_data)} train / {len(test_data)} test")

    base_predicates = PredicateBase()
    names = predicate_names()

    # Build model (DNF by default; pix modes require δSFA_neuralDNF.py)
    if args.guard == "dnf":
        model = NeSyDFA(n=4, m=22, R=2, allowed_edges=allowed, accept_state=3).to(device)
    else:
        assert HAVE_PIX, "NeSyDFA_Pix not found — make sure δSFA_neuralDNF.py is importable."
        use_or_unit = (args.guard == "pix")
        model = NeSyDFA_Pix(
            n=4, m=22, R=2,
            allowed_edges=allowed,
            accept_state=3,
            use_or_unit=use_or_unit,
            slope_and=1.5, slope_or=1.5
        ).to(device)
        # NEW: initialize pix bias-strength (how strongly AND/OR behave as Boolean gates)
        model.set_pix_bias_strength(args.pix_bias_start)

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)  # you used 0.03
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # for BS=1024

    # ------------------------------------------------------------------
    # Regularizers (explanatory comments)
    # ------------------------------------------------------------------
    # lambda_ent : row-entropy weight.
    #   Encourages decisive rows. For a row distribution p(.), entropy H(p) is large when
    #   mass is spread across many edges; minimizing H pushes rows towards near one-hot.
    lambda_ent   = 1e-2

    # lambda_sel : selector sparsity weight.
    #   Encourages *compact* DNFs (few literals per clause; few active clauses).
    #   DNF: penalize 1 − P(IGNORE) for literals + σ(a_r) for clauses.
    #   pix:  L1 on AND weights and (if hybrid) σ(a_r).
    lambda_sel   = 1e-3
    if args.guard in ("pix", "pix_hybrid"):
        lambda_sel *= 0.05  # was 0.2; 0.05 is safer for pix  # 0.2  # softer early sparsity for pix-style units

    # lambda_flow : flow concentration weight (from CLI).
    #   For each time t, define flow_t = α_t ⊗ row(M_t) (broadcast multiply).
    #   We minimize E_t[ −||flow_t||_2^2 ], which discourages spreading flow thinly
    #   across many edges and nudges the model to commit flow to few edges.
    lambda_flow  = args.lambda_flow

    # lambda_part : partition (mutual exclusion) weight (from CLI).
    #   Encourages self vs forward complementarity at row-level *guard* scores g:
    #   (g_self + g_fwd − 1)^2 + ξ·(g_self·g_fwd) with ξ=args.part_xor≥0.
    #   Intuition: either self-loop or forward edge should dominate a row’s guard mass.
    lambda_part  = args.lambda_part
    part_xor     = args.part_xor

    # lambda_prog : progress alignment weight (from CLI).
    #   Optional prior that aligns advances (0→1,1→2,2→3) to weak event sketches G1,G2,G3.
    #   Set to 0 to disable any use of sketches (purely data-driven learning).
    lambda_prog  = args.lambda_prog

    # Temperatures (train vs eval)
    row_temperature_train   = args.tau_row
    literal_temperature_train = args.tau_gs
    row_temperature_eval    = 0.3
    literal_temperature_eval  = 0.7

    # --------------------------
    # Evaluation loss (val BCE)
    # --------------------------
    def eval_loss() -> float:
        """
        Compute validation BCE with fixed (colder) temperatures.
        """
        model.eval()
        total_loss, total_examples = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
                xb = xb.to(device); yb = yb.to(device)
                predicate_truths_seq = u_from_probs(base_predicates, xb)
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
    print(f"[INFO] λ_flow={lambda_flow:.2e}  λ_part={lambda_part:.2e}  part_xor={part_xor:.2f}  "
          f"λ_prog={lambda_prog:.2e}  τ_row={row_temperature_train:.2f}  τ_gs={literal_temperature_train:.2f}")

    def _anneal_tau_row(epoch_idx: int, num_epochs_minus1: int,
                        tau0: float, tau1: float, schedule: str = "linear") -> float:
        """Linear or cosine anneal for the row-softmax temperature over epochs."""
        if schedule == "linear":
            s = epoch_idx / max(1, num_epochs_minus1)
            return tau0 + (tau1 - tau0) * s
        # cosine
        s = epoch_idx / max(1, num_epochs_minus1)
        return tau1 + 0.5 * (tau0 - tau1) * (1 + math.cos(math.pi * s))

    for epoch in range(1, args.epochs + 1):
        # (A) Anneal row temperature across epochs for stable optimization on long sequences
        row_temperature_train = _anneal_tau_row(epoch - 1, args.epochs - 1,
                                                args.tau_row_start, args.tau_row_final, args.tau_sched)
                                                
        # (A.1) NEW: Anneal pix logical-bias strength so gates harden gradually
        if args.guard in ("pix", "pix_hybrid"):
            pix_bias_strength = _anneal_tau_row(epoch - 1, args.epochs - 1,
                                                args.pix_bias_start, args.pix_bias_final, args.pix_bias_sched)
            # This tightens the AND/OR gate thresholds in the pix units
            model.set_pix_bias_strength(pix_bias_strength)                                        
                                                
        model.train()
        epoch_loss_sum, epoch_examples = 0.0, 0

        for batch in train_loader:
            xb, yb = (batch_ts_to_xy(batch) if isinstance(batch, list) else batch)
            xb = xb.to(device); yb = yb.to(device)

            # (1) Map digit probabilities → predicate truths for the whole sequence
            predicate_truths_seq = u_from_probs(base_predicates, xb)  # (B,T,22)

            # (2) Forward DP with traces for auxiliary losses
            #     α_t : beliefs; M_t : transitions; g_t : raw guard strengths per edge
            y_pred, state_beliefs, transition_matrices, guard_strengths = \
                model.forward_with_traces_and_g(predicate_truths_seq,
                                                row_temperature=row_temperature_train,
                                                literal_temperature=literal_temperature_train)

            #------------------------------------------------------------------------------
            # Debug stalling in long horizons
            # -----------------------------------------------------------------------------
            # -----------------------------------------------------------------------------
            # Diagnostic: expected advances per time step (adv_by_time), total expected
            # advances per sequence (EA_batch), and how *spread out in time* the advances
            # are (time-entropy = timeH).
            #
            # Notation:
            #   - alpha_t := state_beliefs[t] ∈ R^{B×n} is the belief (probability) over
            #     states after consuming t symbols; rows sum to 1.
            #   - M_t ∈ R^{B×n×n} is the row-stochastic transition matrix at time t.
            #   - We call an "advance" a forward move q→q+1 for q∈{0,1,2}; i.e., 0→1,1→2,2→3.
            #
            # adv_by_time:
            #   For each time t and sequence b, the expected probability of advancing at t is
            #       a_t(b) = alpha_t(b,0)*M_t(b,0,1) + alpha_t(b,1)*M_t(b,1,2) + alpha_t(b,2)*M_t(b,2,3).
            #   Stacking over t gives adv_by_time ∈ R^{T×B}. Larger values at a time index
            #   mean the model tends to perform a forward move there.
            #
            # EA_batch (Expected Advances per sequence, averaged over batch):
            #   EA_batch = mean_b sum_t a_t(b).
            #   Intuition: how many times, on average, a sequence advances across the whole
            #   run. For the 3-event target, we aim for EA_batch ≈ 3.0.
            #
            # timeH (Time-Entropy of advances, averaged over batch):
            #   For each sequence b, turn its advance curve into a distribution over time:
            #       p_time(:,b) = a_(·)(b) / (sum_t a_t(b) + 1e-6).
            #   Then compute the Shannon entropy in nats:
            #       H_time(b) = - sum_t p_time(t,b) * log p_time(t,b).
            #   Finally average over the batch:
            #       timeH = mean_b H_time(b).
            #
            #   Interpretation:
            #     • Low timeH  → advances are concentrated in a few specific steps
            #                    (peaky timing; effective support size ≈ exp(timeH)).
            #     • High timeH → advances are smeared across many steps (diffuse timing).
            #   Bounds (natural log):
            #     • Minimum ≈ 0 (all advances in one time step).
            #     • Maximum ≈ log(T) when advances are uniformly spread. For T=50,
            #       log(50) ≈ 3.91 nats. Thus timeH≈3.9 ⇒ nearly uniform over time,
            #       timeH≈1.1 ⇒ effective support ≈ exp(1.1) ≈ 3 steps.
            #
            # y_hat_mean:
            #   Mean acceptance probability over the batch at the final time step. Useful
            #   to see if the model is mostly rejecting (≈0), accepting (≈1), or uncertain.
            #
            # Numerical notes:
            #   • We add small epsilons (1e-6, clamp_min) to avoid log(0) and 0/0.
            #   • Shapes: adv_by_time (T,B), p_time (T,B), alpha_t (B,n), M_t (B,n,n).
            # -----------------------------------------------------------------------------

            # Expected advances per sequence (EA) and *when* they happen
            adv_by_time = []  # list of (B,) for each t
            for t, M_t in enumerate(transition_matrices):
                αt = state_beliefs[t]  # (B,n)
                a_t = (αt[:, 0] * M_t[:, 0, 1]  # 0→1
                       + αt[:, 1] * M_t[:, 1, 2]  # 1→2
                       + αt[:, 2] * M_t[:, 2, 3])  # 2→3
                adv_by_time.append(a_t)
            adv_by_time = torch.stack(adv_by_time, 0)  # (T,B)
            EA_batch = adv_by_time.sum(0).mean()  # scalar

            # How concentrated in time are the advances? (time-entropy)
            p_time = adv_by_time / (adv_by_time.sum(0, keepdim=True) + 1e-6)  # (T,B)
            time_entropy = (-(p_time.clamp_min(1e-8).log() * p_time).sum(0)).mean()

            if epoch % 10 == 0:  # or every few epochs
                print(f"[diag] EA≈{EA_batch.item():.2f}  timeH≈{time_entropy.item():.2f}  "
                      f"ŷ_mean={y_pred.mean().item():.3f}")
                      
            if epoch % 10 == 0 and args.guard in ("pix", "pix_hybrid"):
                print(f"[diag] τ_row={row_temperature_train:.2f}  pix_bias={pix_bias_strength:.2f}")          
            # ------------------------------------------------------------------------------
            # Debug stalling in long horizons
            # ------------------------------------------------------------------------------

            # (3) Primary loss: final-time BCE(ŷ, y)
            loss_bce = bce_safe(y_pred, yb)

            # (3b) Any-time acceptance auxiliary (optional; dense gradient across time)
            #      y_t = ⟨α_t, A⟩ (probability of being in accept at time t)
            #      y_any = 1 − ∏_t (1 − y_t)  (probability we were in accept at least once)
            y_t_list = [(alpha_t * model.accept).sum(-1) for alpha_t in state_beliefs[1:]]   # length = T
            y_any = 1.0 - torch.prod(1.0 - torch.stack(y_t_list, dim=0), dim=0)
            loss_any = bce_safe(y_any, yb)

            # (4) Row entropy: encourage decisive (low-entropy) rows
            loss_row_ent = torch.stack([row_entropy(M_t) for M_t in transition_matrices]).mean()

            # (5) Selector sparsity: encourage compact guards (few literals/clauses)
            loss_sel = selector_sparsity_penalty(model)

            # (6) (Optional) Progress alignment using sketches (set λ=0 to disable)
            #     For each t and q∈{0,1,2}, penalize advance q→q+1 when G_{q+1} is *not* active.
            G1, G2, G3 = compute_intended_guards_from_u(predicate_truths_seq)  # (B,T)
            G_events = [G1, G2, G3]
            prog_pen_terms = []
            for t, M_t in enumerate(transition_matrices):
                alpha_t = state_beliefs[t]
                for q in range(3):  # edges 0→1, 1→2, 2→3
                    advance_prob = alpha_t[:, q] * M_t[:, q, q+1]          # (B,)
                    illegal_advance = advance_prob * (1.0 - G_events[q][:, t])
                    prog_pen_terms.append(illegal_advance.mean())
            loss_prog = torch.stack(prog_pen_terms).mean()

            # (7) Flow concentration: nudge expected flow α_t ⊗ M_t to concentrate on few edges
            flow_terms = []
            for t, M_t in enumerate(transition_matrices):
                flow_t = state_beliefs[t].unsqueeze(-1) * M_t            # (B,n,n)
                flow_terms.append(-(flow_t ** 2).mean())
            loss_flow = torch.stack(flow_terms).mean()

            # (8) Row partition: encourage self vs forward mutual exclusivity in raw g
            part_terms = []
            for t, g_raw in enumerate(guard_strengths):
                for q in range(model.n - 1):
                    if torch.isfinite(model.transition.mask[q, q]) and torch.isfinite(model.transition.mask[q, q+1]):
                        g_self = g_raw[:, q, q]
                        g_fwd  = g_raw[:, q, q+1]
                        part_terms.append(((g_self + g_fwd - 1.0)**2 + part_xor * (g_self * g_fwd)).mean())
            loss_part = torch.stack(part_terms).mean() if part_terms else torch.tensor(0.0, device=device)

            # (9) Total loss
            loss = (loss_bce
                    + args.lambda_any * loss_any
                    + lambda_ent   * loss_row_ent
                    + lambda_sel   * loss_sel
                    + lambda_prog  * loss_prog
                    + lambda_flow  * loss_flow
                    + lambda_part  * loss_part
                    )

            if False:  # Doesn't seem to work, learning never progresses in the length-50 case.
            # if args.lambda_contradict > 0.0 or args.lambda_atmost_one > 0.0:
                p_contra, p_am1 = rule_constraint_penalty(model)
                loss = (loss
                        + args.lambda_contradict * p_contra
                        + args.lambda_atmost_one * p_am1)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping helps stability with large LR/BS
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * xb.size(0)
            epoch_examples += xb.size(0)

        train_hist.append(epoch_loss_sum / epoch_examples)
        val_loss = eval_loss()
        val_hist.append(val_loss)
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_hist[-1]:.4f}  val={val_loss:.4f}")

    # --------------------------
    # Curves
    # --------------------------
    plt.figure()
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.title("Training/Validation")
    plt.tight_layout(); plt.savefig("loss_curve.png")

    # Edge usage summary on train set
    summarize_edge_usage(model, base_predicates, train_loader)

    # ---------------------------------------------
    # Extraction thresholds:
    #   ts  (literal threshold)
    #       • DNF guards: threshold on the literal *selector* softmax per predicate.
    #         A literal is included in a clause only if max{P(+),P(−)} ≥ ts.
    #         (Selector is over {IGNORE,+,−}; polarity = argmax over {+,−}.)
    #       • pix / pix_hybrid: interpreted as weight-magnitude threshold for AND units.
    #         A predicate becomes a literal if |w_j| ≥ ts; polarity is sign(w_j).
    #
    #   ta  (clause activation threshold)
    #       • DNF guards: threshold on the clause strength α_r = σ(a_r). Keep clause if α_r ≥ ta.
    #       • pix_hybrid: same (noisy-OR is used); pure pix ignores ta.
    #
    # Typical ranges:  ts ∈ [0.5, 0.8] (higher = fewer literals per clause),
    #                  ta ∈ [0.1, 0.5] (higher = fewer clauses overall).
    # ---------------------------------------------
    dnfs_by_edge = model.transition.extract_all(names, ts=args.extract_th, ta=args.extract_ta)
    print("\nLearned guards (DNFs) per ALLOWED edge (q->q'):\n")
    for (q, qp), clauses in dnfs_by_edge.items():
        if not clauses:
            continue
        print(f"{q}->{qp}:  " + "  OR  ".join(pretty_clause(c) for c in clauses))

    # Optional graph viz
    if HAVE_NX:
        G = nx.DiGraph()
        for s in range(1, 5):
            G.add_node(s, label=f"{s}" + (" (accept)" if s == 4 else ""))
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
