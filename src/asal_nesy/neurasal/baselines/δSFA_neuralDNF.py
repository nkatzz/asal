# ===========================
# pix2rule-style δSFA guards
# ===========================
# What this offers:
#   - SemiSymbolicUnit (pix2rule AND/OR with bias trick + tanh)
#   - SemiSymbolicDNFEdge (R conjunctive units + OR combiner)
#   - TransitionLayerPix (uses SemiSymbolicDNFEdge for all edges)
#   - NeSyDFA_Pix (same API as your NeSyDFA, just a different guard block)
#   - Pruning routines (greedy ε-prune, magnitude threshold, saturate)
#
from typing import List, Tuple, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiSymbolicUnit(nn.Module):
    """
    pix2rule-style semi-symbolic neuron with the bias trick and tanh activation.

    Forward (inputs are in [0,1]):
        x_{±1} = 2 * x_{01} - 1
        s      = <w, x_{±1}>
        β      = δ * (max_i |w_i| - Σ_i |w_i|) / (1 + m)  # lightly normalized
        y      = tanh( slope * ( s + β + b0 ) ) ∈ [-1,1]

    δ:
        +1 → AND-like (needs many high inputs)
        -1 → OR-like  (any high input suffices)
    """
    def __init__(self, in_features: int, delta: float, slope: float = 2.0, init_scale: float = 0.02):
        super().__init__()
        self.in_features = int(in_features)
        self.delta = float(delta)
        self.slope = float(slope)
        # Neutral start: tanh(0)=0 ⇒ (y+1)/2 ≈ 0.5 (good early gradients)
        self.w  = nn.Parameter(torch.zeros(self.in_features))
        self.b0 = nn.Parameter(torch.tensor(0.0))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        xpm1 = x01 * 2.0 - 1.0
        s = (xpm1 * self.w).sum(dim=-1)
        absw = self.w.abs()
        beta_raw = absw.max() - absw.sum()
        beta = self.delta * beta_raw / (1.0 + float(self.in_features))
        y = torch.tanh(self.slope * (s + beta + self.b0))
        return y

    @torch.no_grad()
    def magnitude_mask(self, thresh: float) -> torch.Tensor:
        return self.w.abs() >= thresh

    @torch.no_grad()
    def hard_zero_small(self, thresh: float) -> None:
        mask = self.magnitude_mask(thresh)
        self.w.data = self.w.data * mask

    @torch.no_grad()
    def saturate(self, magnitude: float = 6.0) -> None:
        w = self.w.data
        nz = (w != 0)
        self.w.data = torch.where(nz, magnitude * torch.sign(w), torch.zeros_like(w))

    @torch.no_grad()
    def extract_literals(self, names: List[str], thresh: float = 0.2) -> List[Tuple[str, bool]]:
        lits = []
        for j, name in enumerate(names):
            wj = float(self.w[j].item())
            if abs(wj) >= thresh:
                lits.append((name, wj > 0))
        return lits


class SemiSymbolicDNFEdge(nn.Module):
    """
    An edge guard implemented as R pix2rule AND units (clauses), pooled by OR.

    Modes:
      use_or_unit=True   → OR is pix2rule unit over clause outputs.
      use_or_unit=False  → OR is probabilistic noisy-OR with per-clause α_r.
    """
    def __init__(self, m: int, R: int = 2, slope_and: float = 2.0, slope_or: float = 2.0,
                 use_or_unit: bool = True):
        super().__init__()
        self.m = m
        self.R = R
        self.and_units = nn.ModuleList([SemiSymbolicUnit(m, delta=+1.0, slope=slope_and) for _ in range(R)])
        with torch.no_grad():
            for and_u in self.and_units:
                and_u.w.add_(0.005 * torch.randn_like(and_u.w))
                and_u.b0.add_(0.001 * torch.randn_like(and_u.b0))
        self.use_or_unit = use_or_unit
        if use_or_unit:
            self.or_unit = SemiSymbolicUnit(R, delta=-1.0, slope=slope_or)
            with torch.no_grad():
                self.or_unit.w.copy_(0.05 * torch.randn(R))
                self.or_unit.b0.copy_(torch.tensor(0.01))
        else:
            self.alog = nn.Parameter(torch.zeros(R))

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:
        y_r = [and_u(u_t) for and_u in self.and_units]  # (B,)
        Y = torch.stack(y_r, dim=-1)                    # (B,R) in [-1,1]
        if self.use_or_unit:
            c = (Y + 1.0) * 0.5                         # [0,1]
            y_or = self.or_unit(c)                      # [-1,1]
            g = (y_or + 1.0) * 0.5                      # [0,1]
        else:
            c = (Y + 1.0) * 0.5                         # [0,1]
            alpha = torch.sigmoid(self.alog).unsqueeze(0)  # (1,R)
            g = 1.0 - torch.prod(1.0 - alpha * c, dim=-1)  # (B,)
        return g.clamp(0.0, 1.0)

    @torch.no_grad()
    def magnitude_prune(self, clause_thresh: float = 0.15) -> None:
        for and_u in self.and_units:
            and_u.hard_zero_small(clause_thresh)
        if self.use_or_unit:
            self.or_unit.hard_zero_small(clause_thresh)

    @torch.no_grad()
    def greedy_epsilon_prune(self, eval_fn: Callable[[], float], epsilon: float = 1e-3) -> None:
        def try_prune_weight(wparam: torch.nn.Parameter):
            w = wparam.data
            nz_idx = (w != 0.0).nonzero(as_tuple=True)[0]
            for j in nz_idx.tolist():
                old = float(w[j].item())
                w[j] = 0.0
                new_loss = eval_fn()
                if new_loss > baseline + epsilon:
                    w[j] = old  # revert
        baseline = eval_fn()
        for and_u in self.and_units:
            try_prune_weight(and_u.w)
        if self.use_or_unit:
            try_prune_weight(self.or_unit.w)

    @torch.no_grad()
    def saturate(self, magnitude: float = 6.0) -> None:
        for and_u in self.and_units:
            and_u.saturate(magnitude)
        if self.use_or_unit:
            self.or_unit.saturate(magnitude)

    @torch.no_grad()
    def extract(self, names: List[str], clause_thresh: float = 0.2) -> List[List[Tuple[str, bool]]]:
        dnfs = []
        alpha_keep = None
        if (not self.use_or_unit) and hasattr(self, "alog"):
            alpha = torch.sigmoid(self.alog)  # (R,)
            alpha_keep = (alpha >= clause_thresh).tolist()
        for r, and_u in enumerate(self.and_units):
            if alpha_keep is not None and not alpha_keep[r]:
                continue
            lits = and_u.extract_literals(names, thresh=clause_thresh)
            if lits:
                dnfs.append(lits)
        return dnfs


class TransitionLayerPix(nn.Module):
    """
    Transition layer using SemiSymbolicDNFEdge per allowed edge.

    API parity with DNF TransitionLayer:
      • forward(u_t, row_temperature) → M_t (B,n,n)
      • forward_with_g(u_t, row_temperature) → (M_t, g_t)
      • gates(u_t) → raw g_t for inspection
      • extract_all(...) for rule printing
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int,int]] = None,
                 use_or_unit: bool = True, slope_and: float = 2.0, slope_or: float = 2.0):
        super().__init__(); self.n=n; self.m=m
        self.edges = nn.ModuleList([SemiSymbolicDNFEdge(m, R, slope_and, slope_or, use_or_unit)
                                    for _ in range(n*n)])
        mask = torch.full((n,n), float('-inf'))
        if allowed_edges is None: allowed_edges=[(i,i) for i in range(n)]
        for (q,qp) in allowed_edges: mask[q,qp]=0.0
        self.register_buffer("mask", mask)
        self.logit_scale = 1.0  # NEW: scale for logit(g) → logits

    def _row_fallback(self, logits: torch.Tensor) -> torch.Tensor:
        # NEW: same safety we use in the DNF layer
        B, n, _ = logits.shape
        ok = torch.isfinite(logits).any(dim=-1, keepdim=True)  # (B,n,1)
        if not ok.all():
            eye = torch.eye(n, device=logits.device).unsqueeze(0)
            logits = torch.where(ok, logits, 10.0 * eye)
        return logits

    def forward(self, u_t: torch.Tensor, row_temperature: float = 1.0) -> torch.Tensor:
        """Compute M_t (row-stochastic) from predicate truths u_t."""
        B = u_t.size(0)
        g = torch.stack([e(u_t) for e in self.edges], dim=-1).view(B, self.n, self.n)

        # logits = torch.log(g.clamp_min(1e-6)) + self.mask
        # return F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()

        # CHANGED: logit instead of log (much better gradients near 0.5)
        eps = 1e-6
        logits = self.logit_scale * torch.logit(g.clamp(eps, 1 - eps)) + self.mask
        logits = self._row_fallback(logits)
        return F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()


    def forward_with_g(self, u_t: torch.Tensor, row_temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (M_t, g_t) with the same temperature handling as DNF."""
        g = self.gates(u_t)
        # logits = torch.log(g.clamp_min(1e-6)) + self.mask
        # M_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        # return M_t, g

        eps = 1e-6
        logits = self.logit_scale * torch.logit(g.clamp(eps, 1 - eps)) + self.mask
        logits = self._row_fallback(logits)
        M_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return M_t, g

    def gates(self, u_t: torch.Tensor) -> torch.Tensor:
        """Raw guard scores g(q→q')(t) in [0,1] for all edges."""
        g = torch.stack([e(u_t) for e in self.edges], dim=-1).view(u_t.size(0), self.n, self.n)
        return g.clamp(0.0, 1.0)

    @torch.no_grad()
    def extract_all(self, names, ts: float = 0.2, ta: float = 0.2, clause_thresh: float = None):
        """
        Unified extractor API (compatible with DNF version).
        ts acts as weight-magnitude threshold in pix; ta is used in hybrid via α_r.
        """
        thr = clause_thresh if clause_thresh is not None else ts
        out = {}
        n = self.n
        for q in range(n):
            for qp in range(n):
                if not torch.isfinite(self.mask[q, qp]): continue
                edge = self.edges[q * n + qp]
                edge.magnitude_prune(clause_thresh=thr)
                edge.saturate(magnitude=6.0)
                out[(q + 1, qp + 1)] = edge.extract(names, clause_thresh=thr)
        return out


class NeSyDFA_Pix(nn.Module):
    """
    δSFA with pix2rule-style guards. DP and acceptance API match the DNF model.

    Exposes `self.transition` to keep training utilities (e.g., selector sparsity,
    extraction) unchanged across guard types.
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int,int]] = None,
                 accept_state: int = 3, use_or_unit: bool = True,
                 slope_and: float = 2.0, slope_or: float = 2.0):
        super().__init__(); self.n=n
        self.trans = TransitionLayerPix(n,m,R,allowed_edges,use_or_unit,slope_and,slope_or)
        self.transition = self.trans               # <-- alias for API parity
        self.register_buffer("alpha0", F.one_hot(torch.tensor(0), num_classes=n).float().unsqueeze(0))
        self.register_buffer("accept", F.one_hot(torch.tensor(accept_state), num_classes=n).float())

    def forward(self, u_seq: torch.Tensor,
                row_temperature: float = 1.0,
                literal_temperature: float = 0.0) -> torch.Tensor:
        """
        Match DNF forward signature: returns final acceptance probability ŷ.
        """
        B, T, _ = u_seq.shape
        alpha = self.alpha0.repeat(B, 1)
        for t in range(T):
            M_t = self.transition(u_seq[:, t, :], row_temperature=row_temperature)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
        yhat = (alpha * self.accept).sum(-1).clamp(0.0, 1.0)
        return yhat

    def forward_with_traces_and_g(self,
                                  u_seq: torch.Tensor,
                                  row_temperature: float = 1.0,
                                  literal_temperature: float = 0.0):
        """
        Runs DP and returns acceptance plus per-step traces, with gradients enabled.

        Returns:
            yhat:   (B,)
            alphas: [T+1] of (B, n)        beliefs
            Ms:     [T]   of (B, n, n)     transition matrices
            Gs:     [T]   of (B, n, n)     raw guard strengths
        """
        B, T, _ = u_seq.shape
        alphas, Ms, Gs = [], [], []
        alpha = self.alpha0.repeat(B, 1)
        alphas.append(alpha)
        tr = max(row_temperature, 1e-6)
        for t in range(T):
            u_t = u_seq[:, t, :]
            M_t, g_t = self.transition.forward_with_g(u_t, row_temperature=tr)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
            Gs.append(g_t); Ms.append(M_t); alphas.append(alpha)
        yhat = (alpha * self.accept).sum(-1).clamp(0.0, 1.0)
        return yhat, alphas, Ms, Gs
