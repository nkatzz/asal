#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# sklearn for metrics
from sklearn.metrics import f1_score, confusion_matrix

# ------------------------ External data API (unchanged) ------------------------
try:
    from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader, TensorSequence
    HAVE_EXT = True
except Exception:
    HAVE_EXT = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
np.random.seed(1); random.seed(1); torch.manual_seed(1)

# ------------------------ Predicate bank (minimal non-redundant) ------------------------
class PredicateBase:
    """
    Minimal, non-redundant bank:
      - parity:   even            (odd = NOT even via negative weight)
      - ordinal:  le_0..le_8      (ge_t = NOT le_{t-1}; le_9 is tautology)
    Total per dim: 1 + 9 = 10 predicates.
    """
    def __init__(self):
        self.even_idx = torch.tensor([0,2,4,6,8], dtype=torch.long)

    def compute(self, digit_probs_t: torch.Tensor) -> torch.Tensor:
        # digit_probs_t: (B,10)
        p = digit_probs_t.float()
        even_truth = p.index_select(1, self.even_idx.to(p.device)).sum(1, keepdim=True)  # (B,1)
        cumsum = torch.cumsum(p, dim=1)                                                  # (B,10)
        le_stack = torch.cat([cumsum[:, k:k+1] for k in range(9)], dim=1)                # le_0..le_8 (B,9)
        return torch.cat([even_truth, le_stack], dim=1).float()

def predicate_names_multi(dim_names: List[str], D: int) -> List[str]:
    base = ["even"] + [f"le_{k}" for k in range(9)]  # no odd, no ge_*, no le_9
    out = []
    for d in range(D):
        pref = dim_names[d] if d < len(dim_names) else f"d{d+1}"
        out.extend([f"{pref}_{b}" for b in base])
    return out

def batch_ts_to_xy(batch: List["TensorSequence"], sharp: float = 0.98):
    """Turn a batch of TensorSequence into (B,T,D,10) symbol probabilities + (B,) labels.
       Here we create peaked distributions per label for simplicity; replace with your CNN if needed."""
    assert len(batch) > 0
    ts0 = batch[0]
    T, D = ts0.seq_length, ts0.dimensionality
    # infer names from first step
    names = []
    for d in range(D):
        keys = list(ts0.image_labels[0][d].keys())
        names.append(keys[0] if keys else f"d{d+1}")
    def one_probs(ts):
        grid = np.zeros((T, D), np.int32)
        for t in range(T):
            for d in range(D):
                grid[t, d] = int(next(iter(ts.image_labels[t][d].values())))
        out = np.empty((T, D, 10), np.float32)
        for t in range(T):
            for d in range(D):
                probs = np.ones(10, np.float32) * (1 - sharp) / 9.0
                probs[grid[t, d]] = sharp
                out[t, d] = probs / probs.sum()
        return out, int(ts.seq_label)
    probs_list, y_list = [], []
    for ts in batch:
        p, y = one_probs(ts)
        probs_list.append(p); y_list.append(y)
    xb = torch.from_numpy(np.stack(probs_list, 0)).float()  # (B,T,D,10)
    yb = torch.tensor(y_list, dtype=torch.float32)
    return xb, yb, names

def u_from_probs_any(bank: PredicateBase, prob_seq: torch.Tensor) -> torch.Tensor:
    """(B,T,10) -> (B,T,10); (B,T,D,10) -> (B,T,10*D)  (10 per dim with minimal bank)"""
    if prob_seq.dim() == 3:
        B, T, _ = prob_seq.size()
        out = torch.stack([bank.compute(prob_seq[:, t, :]) for t in range(T)], dim=1)   # (B,T,10)
        return out
    elif prob_seq.dim() == 4:
        B, T, D, _ = prob_seq.size()
        per_t = []
        for t in range(T):
            per_d = [bank.compute(prob_seq[:, t, d, :]) for d in range(D)]
            per_t.append(torch.cat(per_d, dim=1))                                       # (B,10*D)
        return torch.stack(per_t, dim=1)                                                # (B,T,10*D)
    else:
        raise ValueError(f"Bad prob_seq shape: {tuple(prob_seq.shape)}")

# ------------------------ Loss helpers ------------------------
def bce_safe(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    y_pred = torch.nan_to_num(y_pred, nan=0.5, posinf=1-1e-6, neginf=1e-6).clamp(1e-6, 1-1e-6)
    y_true = y_true.clamp(0.0, 1.0)
    return F.binary_cross_entropy(y_pred, y_true)

# ------------------------ Hard-Concrete (Louizos) ------------------------
class HardConcrete(nn.Module):
    """
    Hard-Concrete gate (Louizos et al., 2018) for L0 regularization.
    Produces z in [0,1] with reparameterization; expected L0 has closed form.
    """
    def __init__(self, shape, beta: float = 2./3., gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(shape))  # init near 0.5 open prob
        self.beta = beta; self.gamma = gamma; self.zeta = zeta

    def _stretch(self, s):
        return s * (self.zeta - self.gamma) + self.gamma

    def sample(self, training: bool, device=None):
        if training:
            u = torch.rand_like(self.log_alpha, device=self.log_alpha.device if device is None else device)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha / self.beta)
        z = torch.clamp(self._stretch(s), 0.0, 1.0)
        return z

    def expected_L0(self):
        # E[gate > 0] = sigmoid(log_alpha - beta * log(-gamma/zeta))
        thresh = -self.gamma / self.zeta
        return torch.sigmoid(self.log_alpha - self.beta * torch.log(torch.tensor(thresh, device=self.log_alpha.device)))

# ------------------------ Guard modules ------------------------
class ClauseCombiner:
    @staticmethod
    def combine(u_t: torch.Tensor, w_eff: torch.Tensor, combiner: str, kappa: float, eps: float = 1e-6):
        """
        u_t: (B,m) predicate truths in [0,1]
        w_eff: (m,) or (B,m) effective signed weights per predicate (per-guard)
        Returns: (B,) clause value in [0,1]
        """
        u = u_t.clamp(eps, 1.0 - eps)
        if w_eff.dim() == 1:
            w = w_eff.unsqueeze(0).expand(u.size(0), -1)
        else:
            w = w_eff

        if combiner == "logit-sum":
            logits = torch.log(u) - torch.log(1 - u)               # logit(u)
            s = (w * logits).sum(dim=-1)                           # (B,)
            return torch.sigmoid(kappa * s)

        elif combiner == "product":
            # positive & negative parts act as exponents on u and (1-u)
            w_pos = torch.clamp(w, min=0.0)
            w_neg = torch.clamp(-w, min=0.0)
            log_c = (w_pos * torch.log(u) + w_neg * torch.log(1 - u)).sum(dim=-1)
            return torch.exp(log_c).clamp(0.0, 1.0)

        else:
            raise ValueError(f"Unknown combiner: {combiner}")

class Guard(nn.Module):
    """
    One guard (clause) on a given edge.
    - local signed weights v_i  (tanh to [-1,1])
    - optional global predicate weights z_i (sigmoid or Hard-Concrete gate)
    Effective weight: w_i = (global_gate or 1) * tanh(v_i)
    """
    def __init__(self, m: int, weighting: str, combiner: str, kappa: float,
                 reg_type: str, global_gate: Optional[HardConcrete|nn.Parameter]):
        super().__init__()
        assert weighting in ("local","hybrid")
        assert combiner in ("logit-sum","product")
        assert reg_type in ("l1","l0")
        self.m = m
        self.weighting = weighting
        self.combiner = combiner
        self.kappa = kappa
        self.reg_type = reg_type
        self.v = nn.Parameter(torch.zeros(m))          # signed pre-weights
        nn.init.normal_(self.v, mean=0.0, std=0.02)
        self.global_gate = global_gate                 # None for 'local' mode or if not used

    def effective_w(self, training: bool) -> torch.Tensor:
        w = torch.tanh(self.v)                         # [-1,1]
        if self.global_gate is not None:
            if isinstance(self.global_gate, HardConcrete):
                z = self.global_gate.sample(training=training)      # [0,1]
            else:
                z = torch.sigmoid(self.global_gate)                 # [0,1]
            w = z * w
        return w

    def forward(self, u_t: torch.Tensor, training: bool) -> torch.Tensor:
        w = self.effective_w(training=training)        # (m,)
        return ClauseCombiner.combine(u_t, w, self.combiner, self.kappa)

    # NEW: pre-sigmoid clause score for OR=maxpool
    def score(self, u_t: torch.Tensor, training: bool) -> torch.Tensor:
        """
        Returns s_r (pre-sigmoid). For product combiner: s_r = logit(c_r).
        For logit-sum: s_r = kappa * sum_i w_i * logit(u_i).
        """
        u = u_t.clamp(1e-6, 1 - 1e-6)
        w = self.effective_w(training=training)
        if self.combiner == "logit-sum":
            logits = torch.log(u) - torch.log(1 - u)            # logit(u)
            s = (w * logits).sum(dim=-1)
            return self.kappa * s                                # (B,)
        else:  # product
            w_pos = torch.clamp(w, min=0.0)
            w_neg = torch.clamp(-w, min=0.0)
            log_c = (w_pos * torch.log(u) + w_neg * torch.log(1 - u)).sum(dim=-1)
            return self.kappa * log_c  # use log-prob as score; keeps s≈0 initially

    # regularization of predicate weights (per guard)
    def pred_L1(self) -> torch.Tensor:
        w = torch.tanh(self.v)
        if self.global_gate is not None:
            if isinstance(self.global_gate, HardConcrete):
                # with L0 we count gates separately; do not add L1 here
                return torch.zeros((), device=self.v.device)
            z = torch.sigmoid(self.global_gate)
            w = z * w
        return w.abs().sum()

    def pred_L0(self) -> torch.Tensor:
        if self.global_gate is not None and isinstance(self.global_gate, HardConcrete):
            return self.global_gate.expected_L0().sum()
        return torch.zeros((), device=self.v.device)

class NeuralDNFGuard(nn.Module):
    """
    Up to R guards (disjuncts) on an SFA transition.
    OR aggregation:
      - 'noisyor' : original noisy-OR with guard weights (sigmoid(guard_logit))
      - 'maxpool' : smooth max via log-sum-exp of per-clause scores + per-clause biases, then sigmoid
    """
    def __init__(self, m: int, R: int, weighting: str, combiner: str, kappa: float,
                 reg_type: str, share_global: Optional[HardConcrete|nn.Parameter],
                 or_mode: str = "maxpool", beta: float = 8.0):
        super().__init__()
        assert or_mode in ("noisyor", "maxpool")
        self.R = R
        self.or_mode = or_mode
        self.beta = beta
        self.guards = nn.ModuleList([
            Guard(m, weighting, combiner, kappa, reg_type, global_gate=share_global)
            for _ in range(R)
        ])
        # noisy-OR params (kept for backward-compat/extraction)
        self.guard_logit = nn.Parameter(torch.zeros(R))  # α_r (after sigmoid) used only in noisy-OR
        # max-pool bias per clause
        self.bias = nn.Parameter(torch.zeros(R))         # b_r added to score in maxpool

    def forward(self, u_t: torch.Tensor, training: bool) -> torch.Tensor:
        if self.or_mode == "maxpool":
            # stack clause scores (pre-sigmoid), add bias, aggregate with LSE
            S = torch.stack([g.score(u_t, training=training) for g in self.guards], dim=-1)  # (B,R)
            S = S + self.bias.unsqueeze(0)
            s = (1.0/self.beta) * torch.logsumexp(self.beta * S, dim=-1)                     # (B,)
            return torch.sigmoid(s)
        else:
            clause_vals = torch.stack([g(u_t, training=training) for g in self.guards], dim=-1)  # (B,R)
            alpha = torch.sigmoid(self.guard_logit).unsqueeze(0)                                  # (1,R)
            g_t = 1.0 - torch.prod(1.0 - alpha * clause_vals, dim=-1)                            # (B,)
            return g_t

    # regularizers
    def reg_pred_L1(self) -> torch.Tensor:
        return sum(g.pred_L1() for g in self.guards)

    def reg_pred_L0(self) -> torch.Tensor:
        return sum(g.pred_L0() for g in self.guards)

    def reg_guard_L1(self) -> torch.Tensor:
        # L1 on noisy-OR α (minor), plus L1 on maxpool biases (main pruning)
        reg = torch.sigmoid(self.guard_logit).abs().sum()
        reg = reg + self.bias.abs().sum()
        return reg

# ------------------------ Transition and DFA ------------------------
def build_transitions(num_states: int,
                      allow_backward: bool = False,
                      allow_jumps: bool = False) -> List[Tuple[int, int]]:
    """
    Create a list of transitions for an automaton with states numbered 0..num_states-1.

    Specs:
      - States: 0 is the start; num_states - 1 is the accepting state.
      - Always include self-loops (i, i) for every state i.
      - Always include forward step transitions (i, i+1) where valid.
      - If allow_jumps is True, include forward jumps (i, j) for j >= i+2.
      - If allow_backward is True, include backward transitions EXCEPT from the accepting state:
          * Backward step (i, i-1) for i > 0 and i != accepting
          * If allow_jumps is also True, include backward jumps (i, j) for j <= i-2 and i != accepting
      - No backward transition is allowed *from* the accepting state.
    """
    from itertools import permutations
    if num_states < 1:
        raise ValueError("num_states must be >= 1")

    accepting = num_states - 1
    self_loops = [(i, i) for i in range(num_states)]
    step_transitions = [(i, i+1) for i in range(num_states-1)]
    transitions = step_transitions + self_loops
    if allow_backward:
        transitions = transitions +  [(i, j) for i, j in list(permutations(range(num_states), 2)) if i > j and i != accepting]
    if allow_jumps:
        transitions = transitions +  [(i, j) for i, j in list(permutations(range(num_states), 2)) if j > i+1]
    return transitions

class TransitionLayer(nn.Module):
    def __init__(self, n: int, m: int, R: int,
                 allowed_edges: List[Tuple[int,int]],
                 weighting: str, combiner: str, kappa: float, reg_type: str,
                 self_complement: bool,
                 or_mode: str = "maxpool", beta: float = 8.0):
        super().__init__()
        self.n, self.m, self.R = n, m, R
        self.allowed = allowed_edges
        self.self_complement = self_complement
        self.or_mode = or_mode
        self.beta = beta

        # mask of allowed edges
        mask = torch.full((n, n), float('-inf'))
        for (i,j) in allowed_edges:
            mask[i,j] = 0.0
        self.register_buffer("mask", mask)

        # global predicate weight/gate for 'hybrid'
        if weighting == "hybrid":
            if reg_type == "l0":
                self.global_gate = HardConcrete(m)                 # z_i in [0,1]
            else:
                self.global_gate = nn.Parameter(torch.zeros(m))    # z_i = sigmoid(param)
        else:
            self.global_gate = None

        # one module per edge (including self) in n*n order; unused entries = Identity
        self.edges = nn.ModuleList()
        for i in range(n):
            for j in range(n):
                if not torch.isfinite(self.mask[i,j]):
                    self.edges.append(nn.Identity())
                else:
                    self.edges.append(NeuralDNFGuard(
                        m=m, R=R, weighting=weighting, combiner=combiner, kappa=kappa,
                        reg_type=reg_type, share_global=self.global_gate,
                        or_mode=self.or_mode, beta=self.beta
                    ))

    def forward(self, u_t: torch.Tensor, row_tau: float, training: bool) -> torch.Tensor:
        # u_t: (B,m)
        B = u_t.size(0)
        # First compute all non-self edge guards normally
        gmat = torch.zeros(B, self.n, self.n, device=u_t.device)
        for i in range(self.n):
            for j in range(self.n):
                k = i*self.n + j
                if not torch.isfinite(self.mask[i,j]):
                    continue
                if self.self_complement and i == j:
                    # defer self-loop; compute after we have all non-self guards for row i
                    continue
                mod = self.edges[k]
                if isinstance(mod, NeuralDNFGuard):
                    gmat[:, i, j] = mod(u_t, training=training)
                else:
                    gmat[:, i, j] = 0.0

        # Now fill self-loops as complement of all outgoing (excluding self):
        if self.self_complement:
            for i in range(self.n):
                leave = 1.0 - torch.prod(1.0 - gmat[:, i, :] + 1e-6, dim=-1)  # (B,)
                gmat[:, i, i] = (1.0 - leave).clamp(0.0, 1.0)

        # Row softmax over logits
        eps = 1e-6
        logits = torch.logit(gmat.clamp(eps, 1 - eps)) + self.mask
        row_has = torch.isfinite(logits).any(dim=-1, keepdim=True)
        if not row_has.all():
            eye = torch.eye(self.n, device=logits.device).unsqueeze(0)
            logits = torch.where(row_has, logits, 10.0 * eye)
        M_t = F.softmax(logits / max(row_tau, 1e-6), dim=-1)
        return M_t, gmat

    # regularization aggregators
    def reg_pred_L1(self) -> torch.Tensor:
        tot = torch.zeros((), device=self.mask.device)
        for mod in self.edges:
            if isinstance(mod, NeuralDNFGuard):
                tot = tot + mod.reg_pred_L1()
        return tot

    def reg_pred_L0(self) -> torch.Tensor:
        tot = torch.zeros((), device=self.mask.device)
        if isinstance(self.global_gate, HardConcrete):
            tot = tot + self.global_gate.expected_L0().sum()
        for mod in self.edges:
            if isinstance(mod, NeuralDNFGuard):
                tot = tot + mod.reg_pred_L0()
        return tot

    def reg_guard_L1(self) -> torch.Tensor:
        tot = torch.zeros((), device=self.mask.device)
        for mod in self.edges:
            if isinstance(mod, NeuralDNFGuard):
                # includes bias L1 as well
                tot = tot + mod.reg_guard_L1()
        return tot

class NeSySFA(nn.Module):
    def __init__(self, n: int, m: int, R: int,
                 allowed_edges: List[Tuple[int,int]],
                 weighting: str, combiner: str, kappa: float, reg_type: str,
                 self_complement: bool,
                 or_mode: str, beta: float):
        super().__init__()
        self.n = n
        self.transition = TransitionLayer(n, m, R, allowed_edges,
                                          weighting, combiner, kappa, reg_type,
                                          self_complement=self_complement,
                                          or_mode=or_mode, beta=beta)
        self.register_buffer("alpha0", F.one_hot(torch.tensor(0), num_classes=n).float().unsqueeze(0))
        self.register_buffer("accept", F.one_hot(torch.tensor(n-1), num_classes=n).float())

    def forward(self, u_seq: torch.Tensor, row_tau: float) -> torch.Tensor:
        B, T, _ = u_seq.shape
        alpha = self.alpha0.repeat(B, 1)
        for t in range(T):
            M_t, _ = self.transition(u_seq[:, t, :], row_tau=row_tau, training=self.training)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
        y = (alpha * self.accept).sum(-1)
        return y

    def forward_with_traces(self, u_seq: torch.Tensor, row_tau: float):
        B, T, _ = u_seq.shape
        alpha = self.alpha0.repeat(B, 1)
        Ms, Gs = [], []
        for t in range(T):
            M_t, g_t = self.transition(u_seq[:, t, :], row_tau=row_tau, training=self.training)
            Ms.append(M_t); Gs.append(g_t); alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
        y = (alpha * self.accept).sum(-1)
        return y, Ms, Gs

# ------------------------ Extraction utilities ------------------------
@torch.no_grad()
def extract_guards(model: NeSySFA, names: List[str],
                   topk: Optional[int] = None, thresh: Optional[float] = None) -> List[str]:
    """
    Return a list of human-readable guard descriptions per edge.
    For each guard (clause): keep either the top-k predicates by |w| or those with |w|>=thresh.
    Sign of w encodes POS/NEG.
    """
    assert topk is not None or thresh is not None, "Provide topk or thresh for extraction."
    lines = []
    n = model.n
    idx = 0
    for i in range(n):
        for j in range(n):
            mod = model.transition.edges[idx]; idx += 1
            if not isinstance(mod, NeuralDNFGuard):
                continue
            # collect per-guard literals
            if mod.or_mode == "maxpool":
                # show sigmoid(bias) as indicative strength
                strengths = torch.sigmoid(mod.bias).detach().cpu().numpy()
            else:
                strengths = torch.sigmoid(mod.guard_logit).detach().cpu().numpy()
            for r, guard in enumerate(mod.guards):
                w = guard.effective_w(training=False).detach().cpu()  # (m,)
                if topk is not None:
                    k = min(topk, w.numel())
                    _, ids = torch.topk(w.abs(), k)
                    ids = ids.tolist()
                else:
                    mask = (w.abs() >= float(thresh)).nonzero(as_tuple=False).view(-1)
                    ids = mask.tolist()
                if len(ids) == 0 or strengths[r] <= 1e-6:
                    continue
                lits = []
                for pid in ids:
                    sign = "+" if w[pid].item() > 0 else "-"
                    lits.append(f"{names[pid]}{sign}")
                lines.append(f"{i}->{j}  (strength={strengths[r]:.3f}):  " + ", ".join(lits))
    return lines

# ------------------------ Training/eval ------------------------
@torch.no_grad()
def evaluate_metrics(model: NeSySFA, bank: PredicateBase, loader: DataLoader,
                     row_tau_eval: float = 0.3):
    model.eval()
    ys, yh = [], []
    for batch in loader:
        xb, yb, _ = batch_ts_to_xy(batch)
        xb, yb = xb.to(device), yb.to(device)
        u = u_from_probs_any(bank, xb)
        y = model(u, row_tau=row_tau_eval)
        yh.append((y > 0.5).float().cpu().numpy())
        ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys, 0).astype(np.int32)
    y_pred = np.concatenate(yh, 0).astype(np.int32)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return f1, cm

@torch.no_grad()
def evaluate_bce(model: NeSySFA, bank: PredicateBase, loader: DataLoader, row_tau_eval: float = 0.3):
    model.eval()
    tot, N = 0.0, 0
    for batch in loader:
        xb, yb, _ = batch_ts_to_xy(batch)
        xb, yb = xb.to(device), yb.to(device)
        u = u_from_probs_any(bank, xb)
        y = model(u, row_tau=row_tau_eval)
        tot += bce_safe(y, yb).item() * xb.size(0); N += xb.size(0)
    return tot / max(1, N)

def main():
    ap = argparse.ArgumentParser("δSFA (clean + MaxPool-OR): product/logit combiners, optional self-loop complement, metrics & extraction.")
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--test-path", required=True)
    ap.add_argument("--num-states", type=int, default=4)
    ap.add_argument("--num-ors", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--row-tau-start", type=float, default=1.0)
    ap.add_argument("--row-tau-final", type=float, default=0.2)
    ap.add_argument("--tau-sched", choices=["linear","cosine"], default="linear")

    # Design choices
    ap.add_argument("--weighting", choices=["local","hybrid"], default="local",
                    help="Predicate weights are per-guard (local) or global×local (hybrid).")
    ap.add_argument("--combiner", choices=["logit-sum","product"], default="product")
    ap.add_argument("--kappa", type=float, default=3.0, help="Fixed gain for logit-sum (unused for product).")
    ap.add_argument("--reg-type", choices=["l1","l0"], default="l0",
                    help="l1 = L1 on predicate weights; l0 = Louizos hard-concrete expected L0.")
    ap.add_argument("--lambda-pred", type=float, default=0.01, help="Sparsity on predicate weights (L1 or L0-count).")
    ap.add_argument("--lambda-guard", type=float, default=5e-4, help="L1 on guard weights (includes clause bias).")
    ap.add_argument("--self-complement", action="store_true",
                    help="Make self-loops be the complement (noisy-OR) of all outgoing transitions from the same state.")

    # NEW: MaxPool OR
    ap.add_argument("--or-mode", choices=["noisyor","maxpool"], default="maxpool",
                    help="Disjunction aggregation. 'maxpool' = LSE(s_r + b_r) then sigmoid; 'noisyor' = original.")
    ap.add_argument("--beta", type=float, default=15,  # 8.0
                    help="Temperature for log-sum-exp in maxpool (higher = closer to hard max).")
    ap.add_argument("--lambda-bias", type=float, default=0.1,  # 1e-1
                    help="L1 on per-clause bias (prunes unused clauses) in maxpool mode.")

    # Extraction/reporting
    ap.add_argument("--print-guards", action="store_true", help="Print extracted guards at the end.")
    ap.add_argument("--extract-topk", type=int, default=None, help="Top-k literals per guard to print.")
    ap.add_argument("--extract-thresh", type=float, default=None, help="Absolute weight threshold for printing literals.")
    args = ap.parse_args()

    assert HAVE_EXT, "Could not import data utilities (get_data/get_data_loader)."
    train_data, test_data = get_data(args.train_path, args.test_path)
    train_loader = get_data_loader(train_data, args.batch_size, train=True)
    test_loader  = get_data_loader(test_data,  args.batch_size, train=False)

    # Probe one batch to get D and predicate dimension
    bank = PredicateBase()
    xb0, yb0, dim_names = batch_ts_to_xy(next(iter(train_loader)))
    D = xb0.shape[2]
    names = predicate_names_multi(dim_names, D)
    m = len(names)
    print(f"[INFO] D={D}, predicates per dim={m//D}, total m={m}, names[0..3]={names[:4]}")

    # Build δSFA
    allowed = build_transitions(args.num_states, allow_backward=False, allow_jumps=False)
    model = NeSySFA(n=args.num_states, m=m, R=args.num_ors,
                    allowed_edges=allowed,
                    weighting=args.weighting,
                    combiner=args.combiner,
                    kappa=args.kappa,
                    reg_type=args.reg_type,
                    self_complement=args.self_complement,
                    or_mode=args.or_mode, beta=args.beta).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    def anneal_tau(e):
        if args.tau_sched == "cosine":
            s = (e) / max(1, args.epochs-1)
            return args.row_tau_final + 0.5*(args.row_tau_start - args.row_tau_final)*(1+math.cos(math.pi*s))
        s = (e) / max(1, args.epochs-1)
        return args.row_tau_start + (args.row_tau_final - args.row_tau_start) * s

    best_val = float("inf")
    for epoch in range(1, args.epochs+1):
        row_tau = anneal_tau(epoch-1)
        model.train()
        tot_loss, N = 0.0, 0
        for batch in train_loader:
            xb, yb, _ = batch_ts_to_xy(batch)
            xb, yb = xb.to(device), yb.to(device)
            u = u_from_probs_any(bank, xb)

            y, Ms, Gs = model.forward_with_traces(u, row_tau=row_tau)
            loss_bce = bce_safe(y, yb)

            # sparsity regularization (predicate)
            if args.reg_type == "l0":
                reg_pred = model.transition.reg_pred_L0()
            else:
                reg_pred = model.transition.reg_pred_L1()

            # guard weights L1 (includes maxpool bias L1 via EdgeGuards.reg_guard_L1)
            reg_guard = model.transition.reg_guard_L1()

            # scale bias part only with lambda-bias (keep α L1 tiny via lambda-guard)
            # Split the reg into α-part and bias-part:
            reg_alpha, reg_bias = torch.zeros((), device=device), torch.zeros((), device=device)
            for mod in model.transition.edges:
                if isinstance(mod, NeuralDNFGuard):
                    reg_alpha = reg_alpha + torch.sigmoid(mod.guard_logit).abs().sum()
                    reg_bias  = reg_bias  + mod.bias.abs().sum()
            loss = loss_bce + args.lambda_pred*reg_pred + args.lambda_guard*reg_alpha + args.lambda_bias*reg_bias

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            tot_loss += loss_bce.item()*xb.size(0)  # BCE only for epoch print
            N += xb.size(0)

        val_bce = evaluate_bce(model, bank, test_loader, row_tau_eval=0.3)
        print(f"[epoch {epoch:03d}] train_bce={tot_loss/max(1,N):.4f}  val_bce={val_bce:.4f}  row_tau={row_tau:.3f}")
        best_val = min(best_val, val_bce)

    # Final metrics: F1 + confusion matrix
    f1, cm = evaluate_metrics(model, bank, test_loader, row_tau_eval=0.3)
    print(f"\n[F1] {f1:.4f}")
    print("[Confusion matrix]\n", cm)

    # Extraction (optional)
    if args.print_guards:
        topk = args.extract_topk
        thr  = args.extract_thresh
        if topk is None and thr is None:
            topk = 5
        lines = extract_guards(model, names, topk=topk, thresh=thr)
        print("\n[Extracted guards]")
        if not lines:
            print("(none)")
        else:
            for s in lines:
                print("  ", s)

    print(f"\n[DONE] best val BCE: {best_val:.4f}")

if __name__ == "__main__":
    main()
