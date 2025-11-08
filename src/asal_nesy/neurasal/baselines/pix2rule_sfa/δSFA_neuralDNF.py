# ===========================
# pix2rule-style δSFA guards
# ===========================
from typing import List, Tuple, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemiSymbolicUnit(nn.Module):
    """
    pix2rule-style semi-symbolic neuron with the bias trick and tanh activation.

    Inputs are in [0,1]. We map to [-1,1], apply a linear form, then add the
    **logical bias** β = δ * (max|w| - Σ|w|). This implements AND (δ=+1) or OR (δ=-1)
    semantics when tanh saturates. We expose a scalar `bias_strength` so the bias
    can be **annealed** during training (0→1).

        x_pm1 = 2*x - 1
        s     = <w, x_pm1>
        β     = bias_strength * δ * (max|w| - Σ|w|)
        y     = tanh(slope * (s + β)) ∈ [-1,1]
    """
    def __init__(self, in_features: int, delta: float, slope: float = 2.0, init_scale: float = 0.02):
        super().__init__()
        self.in_features = int(in_features)
        self.delta = float(delta)
        self.slope = float(slope)
        # Trainable weights; b0 is kept but **frozen** at 0 to keep strict gate semantics.
        self.w  = nn.Parameter(torch.zeros(self.in_features))
        self.b0 = nn.Parameter(torch.tensor(0.0), requires_grad=False)  # keep for BC but unused
        # NEW: annealable multiplier for the logical bias (starts at 1.0 unless set otherwise)
        self.bias_strength: float = 1.0

    def set_bias_strength(self, s: float) -> None:
        """Public hook: set how strong the logical bias β is (0..1+)."""
        self.bias_strength = float(s)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # Map [0,1] → [-1,1] as in pix2rule
        xpm1 = x01 * 2.0 - 1.0
        s = (xpm1 * self.w).sum(dim=-1)
        # **Faithful** logical bias (NO normalization by 1+m; NO extra learnable bias)
        absw = self.w.abs()
        beta_raw = absw.max() - absw.sum()
        beta = self.bias_strength * self.delta * beta_raw
        y = torch.tanh(self.slope * (s + beta))
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

    use_or_unit=True  → OR is pix2rule unit over clause outputs (strict disjunction).
    use_or_unit=False → OR is probabilistic noisy-OR with per-clause α_r.
    """
    def __init__(self, m: int, R: int = 2, slope_and: float = 2.0, slope_or: float = 2.0,
                 use_or_unit: bool = True):
        super().__init__()
        self.m = m
        self.R = R
        self.and_units = nn.ModuleList([SemiSymbolicUnit(m, delta=+1.0, slope=slope_and) for _ in range(R)])
        with torch.no_grad():
            # Small weight jitter for symmetry breaking; **do not** perturb biases (stay at 0).
            for and_u in self.and_units:
                and_u.w.add_(0.005 * torch.randn_like(and_u.w))
        self.use_or_unit = use_or_unit
        if use_or_unit:
            self.or_unit = SemiSymbolicUnit(R, delta=-1.0, slope=slope_or)
            with torch.no_grad():
                self.or_unit.w.copy_(0.05 * torch.randn(R))
                # Keep OR-bias at 0 and frozen to preserve strict OR semantics.
                self.or_unit.b0.data.zero_()
                self.or_unit.b0.requires_grad_(False)
        else:
            self.alog = nn.Parameter(torch.zeros(R))

    def set_bias_strength(self, s: float) -> None:
        """Propagate the annealed bias strength to all constituent units."""
        for and_u in self.and_units:
            and_u.set_bias_strength(s)
        if self.use_or_unit:
            self.or_unit.set_bias_strength(s)

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
    Adds a setter to propagate pix bias-strength (for annealing).
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
        self.logit_scale = 1.0  # Scale for logit(g) → logits (left as-is)

    def set_bias_strength(self, s: float) -> None:
        """Propagate annealed pix bias strength to all edges and units."""
        for e in self.edges:
            e.set_bias_strength(s)

    def _row_fallback(self, logits: torch.Tensor) -> torch.Tensor:
        B, n, _ = logits.shape
        ok = torch.isfinite(logits).any(dim=-1, keepdim=True)  # (B,n,1)
        if not ok.all():
            eye = torch.eye(n, device=logits.device).unsqueeze(0)
            logits = torch.where(ok, logits, 10.0 * eye)
        return logits

    def forward(self, u_t: torch.Tensor, row_temperature: float = 1.0) -> torch.Tensor:
        B = u_t.size(0)
        g = torch.stack([e(u_t) for e in self.edges], dim=-1).view(B, self.n, self.n)
        eps = 1e-6
        logits = self.logit_scale * torch.logit(g.clamp(eps, 1 - eps)) + self.mask
        logits = self._row_fallback(logits)
        return F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()

    def forward_with_g(self, u_t: torch.Tensor, row_temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.gates(u_t)
        eps = 1e-6
        logits = self.logit_scale * torch.logit(g.clamp(eps, 1 - eps)) + self.mask
        logits = self._row_fallback(logits)
        M_t = F.softmax(logits / max(row_temperature, 1e-6), dim=-1).float()
        return M_t, g

    def gates(self, u_t: torch.Tensor) -> torch.Tensor:
        g = torch.stack([e(u_t) for e in self.edges], dim=-1).view(u_t.size(0), self.n, self.n)
        return g.clamp(0.0, 1.0)

    @torch.no_grad()
    def extract_all(self, names, ts: float = 0.2, ta: float = 0.2, clause_thresh: float = None):
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
    δSFA with pix2rule-style guards. Adds a public setter for bias-strength so the
    main training script can anneal it over epochs.
    """
    def __init__(self, n: int, m: int, R: int = 2, allowed_edges: List[Tuple[int,int]] = None,
                 accept_state: int = 3, use_or_unit: bool = True,
                 slope_and: float = 2.0, slope_or: float = 2.0):
        super().__init__(); self.n=n
        self.trans = TransitionLayerPix(n,m,R,allowed_edges,use_or_unit,slope_and,slope_or)
        self.transition = self.trans               # keep API parity
        self.register_buffer("alpha0", F.one_hot(torch.tensor(0), num_classes=n).float().unsqueeze(0))
        self.register_buffer("accept", F.one_hot(torch.tensor(accept_state), num_classes=n).float())

    def set_pix_bias_strength(self, s: float) -> None:
        """Public hook to anneal pix logical bias strength (0..1+) at runtime."""
        self.transition.set_bias_strength(s)

    def forward(self, u_seq: torch.Tensor,
                row_temperature: float = 1.0,
                literal_temperature: float = 0.0) -> torch.Tensor:
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

