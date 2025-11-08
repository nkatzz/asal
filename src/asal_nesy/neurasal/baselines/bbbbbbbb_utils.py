import  torch
from typing import List, Tuple, Dict
import torch.nn as nn
import torch.nn.functional as F


class SymbolDNFEdge(nn.Module):
    r"""
    Symbol-space DNF guard (DNF over CNN posteriors), with a principled AND:

        Per clause r and dimension d:
          • gate w_{r,d} = σ(γ_{r,d}) ∈ [0,1]   (how much this dim participates)
          • mask m_{r,d} ∈ Δ^{K-1}              (symbol subset for this dim)

        Per-dim OR score:   s_{r,d}(t) = m_{r,d}^T p_{t,d} ∈ [0,1]
        AND across dims:    c_r(t) = exp( Σ_d ẃ_{r,d} · log s_{r,d}(t) )  if Σ_d w_{r,d} > 0
                             c_r(t) = 0                                   otherwise
                            where ẃ_{r,d} = w_{r,d} / (Σ_d w_{r,d})

        Clause strength:    ĉ_r(t) = a_r · c_r(t), with a_r = σ(α_r) (small at init)
        Noisy-OR across r:  g(t) = 1 - ∏_r (1 - ĉ_r(t))

    Key properties:
      • If no dimension is selected, the clause is 0 (fixes the "always-on" pathology).
      • If any selected dimension mismatch (small s), the geometric mean shrinks (AND-like).
      • Within each dimension, masks produce OR over symbols.

    Args
    ----
    num_dims : int
        Number of input streams / dimensions D.
    num_symbols : int
        Number of symbols per dimension K (MNIST: 10).
    R : int
        Max number of clauses in the DNF for this edge.
    init_gate_logit : float
        Initial γ_{r,d}; σ(γ) ≈ gate prior. For D=1, something like +1.0 helps (gate ≈ 0.73).
        For D>1, 0.0 (gate ≈ 0.5) is fine.
    init_clause_logit : float
        Initial α_r; σ(α) ≈ a_r. Using a small value (e.g., -2.0) prevents early saturation.

    Forward
    -------
    symbol_probs_t : (B, D*K) or (B, D, K)
        CNN posteriors for one time step.
    literal_temperature : float
        Temperature for the symbol mask softmax (lower → sharper sets).
    Returns
    -------
    guard : (B,) in [0,1]
    """
    def __init__(
        self,
        num_dims: int,
        num_symbols: int = 10,
        R: int = 2,
        init_gate_logit: float = 0.0,
        init_clause_logit: float = -2.0,
    ):
        super().__init__()
        self.D = int(num_dims)
        self.K = int(num_symbols)
        self.R = int(R)

        # Per-dimension gates γ_{r,d} and per-clause strengths α_r
        self.gate_logit   = nn.Parameter(torch.full((self.R, self.D), float(init_gate_logit)))
        self.clause_logit = nn.Parameter(torch.full((self.R,), float(init_clause_logit)))

        # Symbol masks η_{r,d,k}
        self.mask_logit = nn.Parameter(torch.zeros(self.R, self.D, self.K))

    def forward(self, symbol_probs_t: torch.Tensor, literal_temperature: float = 1.0) -> torch.Tensor:
        eps = 1e-8

        # Shape to (B, D, K)
        if symbol_probs_t.dim() == 2:
            B, DK = symbol_probs_t.shape
            assert DK == self.D * self.K, f"SymbolDNFEdge expected last dim={self.D*self.K}, got {DK}"
            p = symbol_probs_t.view(B, self.D, self.K)
        else:
            p = symbol_probs_t  # (B, D, K)
            B = p.size(0)

        # Masks m_{r,d,k} via (tempered) softmax.
        # (You can swap to sparsemax/entmax later; this version keeps no extra deps.)
        T = max(literal_temperature, 1e-6)
        mask = F.softmax(self.mask_logit / T, dim=-1).unsqueeze(0)  # (1, R, D, K)

        # Per-dimension OR scores s_{r,d}(t) = m^T p
        # p: (B,D,K) → (B,1,D,K) to broadcast; mask: (1,R,D,K)
        s = (mask * p.unsqueeze(1)).sum(dim=-1).clamp(min=eps)  # (B, R, D)

        # Gates and clause strengths
        w = torch.sigmoid(self.gate_logit).unsqueeze(0)  # (1, R, D)
        a = torch.sigmoid(self.clause_logit).view(1, self.R)  # (1, R)

        # Normalized weights over selected dimensions
        sum_w = w.sum(dim=-1, keepdim=True)             # (1, R, 1)
        active = (sum_w > eps).float()                  # (1, R, 1)
        w_norm = w / (sum_w + (1.0 - active) * 1.0)     # avoid NaN when all gates=0; unused when inactive

        # AND via normalized geometric mean over selected dimensions.
        # If no dim selected (active=0), log part is irrelevant; we zero out after exp.
        log_clause = (w_norm * s.log()).sum(dim=-1)     # (B, R)
        clause = torch.exp(log_clause) * active.squeeze(-1)  # (B, R), 0 if no selected dims

        # Optional per-clause strength
        clause = clause * a  # (B, R)

        # Noisy-OR across clauses
        guard = 1.0 - (1.0 - clause).clamp(min=0.0, max=1.0).prod(dim=-1)  # (B,)

        return guard.float()

    @torch.no_grad()
    def extract(self, names: List[str], ts: float = 0.5, ta: float = 0.0):
        """
        Grouped extraction for symbol mode.
        Returns a list of clauses, each is a list of (dim_name, [sym_ids]).
        Clause semantics: ∧_d ( ∨_{k∈S_{r,d}} [dim_is_k] ).
        """
        import re
        D, K = self.D, self.K
        gates = torch.sigmoid(self.gate_logit).cpu().numpy()       # (R,D)
        masks = F.softmax(self.mask_logit, dim=-1).cpu().numpy()   # (R,D,K)

        # Recover per-dim names from 'names' list (assumed block-ordered by dim)
        dim_names = []
        for d in range(D):
            tok = names[d * K]
            m = re.match(r"([^_]+)_is_\d+", tok)
            dim_names.append(m.group(1) if m else f"d{d+1}")

        dnfs = []
        for r in range(self.R):
            clause = []
            if gates[r].sum() < ts:  # no active dims in this clause
                continue
            for d in range(D):
                if gates[r, d] < ts:
                    continue
                syms = [k for k in range(K) if masks[r, d, k] >= ts]
                if syms:
                    clause.append((dim_names[d], syms))
            if clause:
                dnfs.append(clause)
        return dnfs


# ------------------ Symbol-space helpers (for --guard sym_dnf) ------------------

def symbol_names_multi(dim_names: List[str], D: int, K: int = 10) -> List[str]:
    """
    Build names like d1_is_0, ..., d1_is_9, d2_is_0, ..., dD_is_9.
    Used for extraction/printing in symbol-DNF mode.
    """
    out = []
    for d in range(D):
        prefix = dim_names[d] if d < len(dim_names) else f"d{d+1}"
        out.extend([f"{prefix}_is_{k}" for k in range(K)])
    return out

def probs_to_symbol_feats(prob_seq: torch.Tensor) -> torch.Tensor:
    """
    Flatten per-dimension class posteriors into a feature vector per time:

      (B,T,D,10) → (B,T, D*10)  by concatenating dimension blocks.

    No predicate bank is used in --guard sym_dnf mode.
    """
    assert prob_seq.dim() == 4, f"expected (B,T,D,10); got {tuple(prob_seq.shape)}"
    B, T, D, K = prob_seq.shape
    return prob_seq.reshape(B, T, D * K).contiguous()

def pretty_symbol_clause_grouped(clause_grouped):
    """
    clause_grouped: list of (dim_name, [k1,k2,...]) for one clause.
    Render as ∧ across dims, with ∨ within each dim, e.g.:
      d1_is_(0∨2∨4∨6∨8) ∧ d2_is_(1∨3∨5∨7∨9)
    """
    parts = []
    for dim_name, sym_list in clause_grouped:
        if len(sym_list) == 1:
            parts.append(f"{dim_name}_is_{sym_list[0]}")
        else:
            inside = " ∨ ".join(f"{dim_name}_is_{k}" for k in sym_list)
            parts.append(f"({inside})")
    return " ∧ ".join(parts)

