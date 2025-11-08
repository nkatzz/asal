import torch
import math
from typing import Tuple
import torch.nn.functional as F



def count_parameters(module: torch.nn.Module, verbose: bool = True) -> int:
    """
    Count trainable parameters in a PyTorch module.
    If verbose=True, prints a table per submodule (1st level).
    """
    total = 0
    rows = []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        total += n
        if verbose:
            rows.append((name, n, tuple(p.shape)))
    if verbose:
        print("\n[PARAMS] Trainable parameters by tensor:")
        for name, n, shape in rows:
            print(f"  {name:<40} {n:>10}  shape={shape}")
        print(f"[PARAMS] Total trainable: {total:,}\n")
    return total


def count_by_child(module: torch.nn.Module):
    print("\n[PARAMS] By top-level child:")
    grand_total = 0
    for cname, child in module.named_children():
        s = sum(p.numel() for p in child.parameters() if p.requires_grad)
        grand_total += s
        print(f"  {cname:<24} {s:>10}")
    print(f"[PARAMS] Grand total      {grand_total:>10}\n")


from typing import List, Tuple

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

    Args:
        num_states: number of states (must be >= 1)
        allow_backward: include backward transitions (except from accepting)
        allow_jumps: include non-adjacent transitions in the allowed directions

    Returns:
        List of (i, j) transitions.
    """
    from itertools import combinations, permutations, product, filterfalse
    if num_states < 1:
        raise ValueError("num_states must be >= 1")

    accepting = num_states - 1
    # transitions: List[Tuple[int, int]] = []
    self_loops = [(i, i) for i in range(num_states)]
    step_transitions = [(i, i+1) for i in range(num_states-1)]
    transitions = step_transitions + self_loops
    # transitions = list(permutations(range(num_states), 2))
    if allow_backward:
        transitions = transitions +  [(i, j) for i, j in list(permutations(range(num_states), 2)) if i > j and i != accepting]
    if allow_jumps:
        transitions = transitions +  [(i, j) for i, j in list(permutations(range(num_states), 2)) if j > i+1]
    return transitions



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


def _edge_gates_from_logits(gate_log_alpha: torch.Tensor,
                            kind: str = "hardconcrete",
                            temperature: float = 2./3.,
                            low: float = -0.1, high: float = 1.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministic gates z in [0,1] and the L0-like regularizer term p_open for each edge.

    Args
    ----
    gate_log_alpha : (n,n) raw logits (learnable parameters)
    kind : 'hardconcrete' or 'sigmoid'
        'hardconcrete': forward uses the *mean* gate (sigmoid(log_alpha)) clamped to [0,1],
                        regularizer uses P(z>0) = sigmoid(log_alpha - tau*log(-low/high)).
        'sigmoid':      forward uses sigmoid(log_alpha); regularizer is simply L1 on z.

    Returns
    -------
    z : (n,n) in [0,1]     — deterministic gate used to scale guards in forward
    reg: (n,n) non-negative — per-edge regularizer contribution (to be summed/averaged)
    """
    if kind == "sigmoid":
        z = torch.sigmoid(gate_log_alpha)                     # (n,n)
        reg = z.abs()                                         # L1 surrogate
        return z, reg

    # hard-concrete style
    z = torch.sigmoid(gate_log_alpha)                         # deterministic mean gate
    # Expected L0 term: P(z>0) under the stretched concrete (Louizos et al., 2018)
    # p_open = sigmoid(log_alpha - tau*log(-low/high))
    # with low<0<1<high; default low=-0.1, high=1.1 as in the paper.
    # This term acts like E[1_{z>0}] and is the principled L0-like regularizer.
    const = temperature * math.log(-low / high)
    p_open = torch.sigmoid(gate_log_alpha - const)
    # Clamp z to [0,1] “by construction” (we already keep it in [0,1] via sigmoid; no sample)
    return z, p_open


def l0_edge_penalty(model, kind: str = "hardconcrete") -> torch.Tensor:
    """
    L0-style penalty on edges: average per-row expected #active edges.
    This keeps scale stable as you change the number of allowed edges per row.
    """
    tl = model.transition
    z, reg = edge_gates(tl.gate_log_alpha, tl.mask, kind=kind, sample=False)
    allowed = torch.isfinite(tl.mask).float()
    # per-row expected count/mass
    row_sum = (reg * allowed).sum(dim=-1)                   # (n,)
    # average across rows that have any allowed edges
    row_has = (allowed.sum(dim=-1) > 0).float()
    return (row_sum * row_has).sum() / (row_has.sum() + 1e-8)

def row_budget_penalty(model, k: float = 2.0, kind: str = "hardconcrete") -> torch.Tensor:
    """
    Soft per-row budget: penalize only excess over k.
    """
    tl = model.transition
    _, reg = edge_gates(tl.gate_log_alpha, tl.mask, kind=kind, sample=False)
    allowed = torch.isfinite(tl.mask).float()
    row_sum = (reg * allowed).sum(dim=-1)                   # (n,)
    excess = F.relu(row_sum - k)
    row_has = (allowed.sum(dim=-1) > 0).float()
    return (excess * row_has).sum() / (row_has.sum() + 1e-8)




def edge_gates(
    gate_log_alpha: torch.Tensor,
    allowed_mask: torch.Tensor,
    *,
    kind: str = "hardconcrete",
    sample: bool = False,
    temperature: float = 2.0/3.0,
    low: float = -0.1,
    high: float = 1.1,
    rng: torch.Generator | None = None,
):
    """
    Compute per-edge gates z ∈ [0,1] and per-edge regularizer terms reg, restricted to allowed edges.

    Args
    ----
    gate_log_alpha : (n,n) learnable logits.
    allowed_mask   : (n,n) bool/float; True/1 for allowed edges.
    kind           : 'hardconcrete' (L0-like) or 'sigmoid' (L1 surrogate).
    sample         : if True and kind='hardconcrete', draw a stretched-concrete sample (train-time);
                     otherwise use the mean (eval-time).
    temperature    : concrete temperature τ.
    low, high      : stretch to (low, high), as in Louizos et al. 2018.
    rng            : optional torch.Generator for reproducible sampling.

    Returns
    -------
    z   : (n,n) gates in [0,1], zero on disallowed edges.
    reg : (n,n) per-edge regularizer contribution:
          • hardconcrete: P(z>0) = sigmoid(log_alpha - τ*log(-low/high))
          • sigmoid     : |z| (L1 surrogate)
    """
    # Zero out disallowed edges consistently
    allowed = torch.isfinite(allowed_mask) if allowed_mask.dtype.is_floating_point else allowed_mask.bool()
    allowed_f = allowed.to(gate_log_alpha.dtype)

    if kind == "sigmoid":
        z = torch.sigmoid(gate_log_alpha)
        z = z * allowed_f
        reg = z.abs() * allowed_f
        return z, reg

    # hard-concrete
    # Expected L0: P(z>0) (used in the regularizer)
    const = temperature * math.log(-low / high)
    p_open = torch.sigmoid(gate_log_alpha - const) * allowed_f  # (n,n)

    if sample:
        # Reparameterized sample u ~ Uniform(0,1) → s = sigmoid((logit u + log_alpha)/τ)
        if rng is None:
            u = torch.rand_like(gate_log_alpha)
        else:
            u = torch.rand_like(gate_log_alpha, generator=rng)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + gate_log_alpha) / temperature)
        z = s * (high - low) + low                          # stretch
        z = z.clamp(0.0, 1.0)                               # project to [0,1]
    else:
        # Deterministic mean gate (no sampling)
        s = torch.sigmoid(gate_log_alpha)
        z = s * (high - low) + low
        z = z.clamp(0.0, 1.0)

    z = z * allowed_f
    return z, p_open





if __name__ == "__main__":
    t = build_transitions(5, allow_backward=True, allow_jumps=True)
    print(t, len(t))