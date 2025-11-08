#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Symbol-DNF δSFA (clean, TensorSequence-ready)

This script learns SFA guards directly in *symbol space* (e.g., digits) using:
  • Conjunction within a clause: product t-norm (implemented as a geometric mean
    over the *selected* dimensions for stability).
  • Disjunction across clauses: noisy-OR (probabilistic co-norm).
  • Row-stochastic transition matrices via masked row-softmax over logit(guard).

Data interface
--------------
Expects your existing loader (from data_structs.py):
    train_data, test_data = get_data(train_pt, test_pt)
    train_loader = get_data_loader(train_data, batch_size, train=True)
    test_loader  = get_data_loader(test_data,  batch_size, train=False)

Each DataLoader batch is a `list[TensorSequence]`. We map that to (x, y):
  x: (B,T,D,K) per-time per-dim symbol distributions (K classes, e.g., 10 digits)
     using, in order of preference:
       1) ts.predicted_softmaxed_seq (T,D,K) if present
       2) one-hot from ts.image_labels[t][d] (digits → one-hots)
  y: (B,) labels in {0,1} from ts.seq_label.

CLI
---
  --train-pt, --test-pt    : required .pt paths
  --states, --clauses      : SFA states (n), DNF clauses per edge (R)
  --epochs, --batch-size, --lr
  --tau-row, --tau-mask    : temperatures (row-softmax, symbol-mask)
  --topology               : 'self_fwd' or 'full'
  --accepting              : accepting state index (default last)
  --device                 : 'cuda' or 'cpu'
  --save, --load           : optional paths to save/load model
  --extract-ts             : threshold for grouped rule extraction

Author: you
"""

from __future__ import annotations
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ccccccccccc_utils import extract_cumulative_mass, extract_topk, evaluate_loader_bce, prune_and_threshold_symbol_dnf


# -----------------------------------------------------------------------------
# Data plumbing for TensorSequence batches → (x,y)
# -----------------------------------------------------------------------------
def batch_to_xy(batch, device, num_classes: int = 10):
    """
    Convert a batch (list[TensorSequence]) into (x, y) for the model.

    Produces:
      x : (B, T, D, K) float32 in [0,1], rows normalized over K.
          If a sequence has `predicted_softmaxed_seq` (T,D,K), we use it.
          Else we build one-hots from symbolic digits in `image_labels[t][d]`.
      y : (B,) float32 in {0,1} from `seq_label`.
    """
    seqs = batch
    B = len(seqs)
    if B == 0:
        raise ValueError("Empty batch from DataLoader.")

    # infer T, D from the first sequence
    T = seqs[0].seq_length
    D = seqs[0].dimensionality

    x = torch.zeros((B, T, D, num_classes), dtype=torch.float32)
    y = torch.zeros((B,), dtype=torch.float32)

    for i, ts in enumerate(seqs):
        y[i] = float(ts.seq_label)

        # build one-hots from image_labels[t][d] (digit → one-hot)
        onehot = torch.zeros((T, D, num_classes), dtype=torch.float32)
        for t in range(T):
            for d in range(D):
                dct = ts.image_labels[t][d]
                if not dct:
                    raise ValueError(f"Empty image_labels at (t={t}, d={d}).")
                # take the first value in the dict (e.g., {'d1': 7} -> 7)
                digit = int(next(iter(dct.values())))
                if not (0 <= digit < num_classes):
                    raise ValueError(f"Digit {digit} not in [0,{num_classes}).")
                onehot[t, d, digit] = 1.0
        x[i] = onehot

    return x.to(device), y.to(device)


# -----------------------------------------------------------------------------
# Pretty-print helpers for grouped symbol-DNFs
# -----------------------------------------------------------------------------
def symbol_names(dim_names: List[str], K: int) -> List[str]:
    """['d1_is_0',...,'d1_is_{K-1}','d2_is_0',...]."""
    out = []
    for dn in dim_names:
        out.extend([f"{dn}={k}" for k in range(K)])
    return out


def pretty_symbol_clause_grouped(clause_grouped: List[Tuple[str, List[int]]]) -> str:
    """
    One clause shown as AND across dims, OR within each dim group.
    E.g., [(d1,[0,2,4]), (d2,[1,3])] → "(d1_is_0 ∨ d1_is_2 ∨ d1_is_4) ∧ (d2_is_1 ∨ d2_is_3)".
    """
    parts = []
    for dim_name, syms in clause_grouped:
        if len(syms) == 1:
            parts.append(f"{dim_name}_is_{syms[0]}")
        else:
            parts.append("(" + " ∨ ".join(f"{dim_name}={k}" for k in syms) + ")")
    return " ∧ ".join(parts)


# -----------------------------------------------------------------------------
# Symbol-DNF guard for a single edge (q→q′)
# -----------------------------------------------------------------------------
class SymbolDNFEdge(nn.Module):
    r"""
    Guard G(t) built from up to R clauses, each clause is an AND across dimensions,
    each dimension contributes an OR over symbols.

    For clause r and dimension d (K symbols):
      • learn mask logits η_{r,d} ∈ R^K → m_{r,d} = softmax(η_{r,d}/τ_mask) ∈ Δ^{K-1}
      • OR score: s_{r,d}(t) = <m_{r,d}, p_{t,d}> ∈ [0,1]

    Soft selection of participating dimensions with π_{r,d} = σ(γ_{r,d}) ∈ [0,1].

    Clause value (product t-norm as geometric mean over selected dims):
      If ∑_d π_{r,d} > 0:
          C_r(t) = exp( ∑_d \tilde π_{r,d} · log s_{r,d}(t) ),  with \tilde π = π / ∑ π
      else:
          C_r(t) = 0

    Noisy-OR across clauses:
      G(t) = 1 - ∏_{r=1..R} (1 - C_r(t))
    """

    def __init__(self, num_dims: int, num_symbols: int, R: int = 1,
                 init_gate_logit: float = 0.0):
        super().__init__()
        self.D = int(num_dims)
        self.K = int(num_symbols)
        self.R = int(R)

        # Per-clause per-dim symbol masks (logits), and per-dim soft selection (gate logits)
        self.mask_logit = nn.Parameter(torch.zeros(self.R, self.D, self.K))
        self.gate_logit = nn.Parameter(torch.full((self.R, self.D), float(init_gate_logit)))

    def forward(self, symbol_probs_t: torch.Tensor, literal_temperature: float = 0.7) -> torch.Tensor:
        """
        Compute guard value for a time-step batch.

        Args
        ----
        symbol_probs_t : (B, D, K) or (B, D*K) symbol distributions per dimension.
        literal_temperature : τ_mask for the per-dim symbol masks.

        Returns
        -------
        guard : (B,) in [0,1]
        """
        eps = 1e-8

        # Force to (B, D, K)
        if symbol_probs_t.dim() == 2:
            B, DK = symbol_probs_t.shape
            assert DK == self.D * self.K, f"Expected {self.D*self.K} last dim, got {DK}."
            p = symbol_probs_t.view(B, self.D, self.K)
        elif symbol_probs_t.dim() == 3:
            B, D, K = symbol_probs_t.shape
            assert D == self.D and K == self.K, f"Expected (B,{self.D},{self.K}), got {symbol_probs_t.shape}."
            p = symbol_probs_t
        else:
            raise ValueError(f"symbol_probs_t shape {tuple(symbol_probs_t.shape)} not supported.")

        # Masks (per (r,d) simplex over K) with temperature
        T = max(float(literal_temperature), 1e-6)
        masks = F.softmax(self.mask_logit / T, dim=-1).unsqueeze(0)        # (1,R,D,K)

        # Per-dim OR scores s_{r,d}(t) = <mask, p>
        s = (masks * p.unsqueeze(1)).sum(dim=-1).clamp(min=eps)             # (B,R,D)

        # Soft dimension selection π_{r,d} ∈ [0,1]
        gates = torch.sigmoid(self.gate_logit).unsqueeze(0)                 # (1,R,D)

        # Normalize gates over dims when any is active
        sum_g = gates.sum(dim=-1, keepdim=True)                             # (1,R,1)
        active = (sum_g > eps).float()                                      # (1,R,1)
        w_norm = torch.where(sum_g > eps, gates / sum_g, torch.zeros_like(gates))  # (1,R,D)

        # Product t-norm as a geometric mean over selected dims
        log_clause = (w_norm * s.log()).sum(dim=-1)                         # (B,R)
        clause = torch.exp(log_clause) * active.squeeze(-1)                 # (B,R)

        # Noisy-OR across clauses
        guard = 1.0 - (1.0 - clause).clamp(0.0, 1.0).prod(dim=-1)           # (B,)
        return guard.float()

    @torch.no_grad()
    def extract(self, names: List[str], ts: float = 0.5) -> List[List[Tuple[str, List[int]]]]:
        """
        Extract a DNF as a list of grouped clauses:
          each clause is [(dim_name, [k1,k2,...]), ...]
        Threshold ts is applied to π_{r,d} (dim selection) and to mask probs over symbols.
        """
        import re
        D, K = self.D, self.K
        gates = torch.sigmoid(self.gate_logit).cpu().numpy()                # (R,D)
        masks = F.softmax(self.mask_logit, dim=-1).cpu().numpy()            # (R,D,K)

        # Infer dimension names from names[d*K] tokens
        dim_names: List[str] = []
        for d in range(D):
            tok = names[d * K]
            m = re.match(r"([^_]+)_is_\d+", tok)
            dim_names.append(m.group(1) if m else f"d{d+1}")

        dnfs: List[List[Tuple[str, List[int]]]] = []
        for r in range(self.R):
            clause = []
            if gates[r].sum() < ts:           # nothing selected in this clause
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


# -----------------------------------------------------------------------------
# Transition layer over all (q→q′) edges + row-softmax with -inf mask
# -----------------------------------------------------------------------------
class TransitionLayer(nn.Module):
    """
    Holds one SymbolDNFEdge per (q→q′). Disallowed edges are -inf masked in row logits.
    """
    def __init__(self, num_states: int, edge_class: type, edge_kwargs: dict,
                 allowed_edges: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.n = int(num_states)
        self.edges = nn.ModuleList([edge_class(**edge_kwargs) for _ in range(self.n * self.n)])

        # Allowed topology mask
        if allowed_edges is None:
            allowed_edges = [(q, q) for q in range(self.n)] + [(q, q + 1) for q in range(self.n - 1)]
        mask = torch.full((self.n, self.n), float("-inf"))
        for (i, j) in allowed_edges:
            if 0 <= i < self.n and 0 <= j < self.n:
                mask[i, j] = 0.0
        self.register_buffer("logit_mask", mask)

    def _edge(self, q: int, qp: int) -> SymbolDNFEdge:
        return self.edges[q * self.n + qp]

    def forward(self, sym_t: torch.Tensor, tau_mask: float = 0.7, tau_row: float = 0.5) -> torch.Tensor:
        """
        Build M_t from guards at time t.

        Args
        ----
        sym_t : (B, D, K) or (B, D*K)
        tau_mask : temperature for symbol masks
        tau_row  : temperature for row-softmax

        Returns
        -------
        M_t : (B, n, n) row-stochastic transitions
        """
        eps = 1e-6
        # compute guard strengths for all edges
        rows = []
        for q in range(self.n):
            row = [self._edge(q, qp)(sym_t, literal_temperature=tau_mask) for qp in range(self.n)]  # list of (B,)
            rows.append(torch.stack(row, dim=1))          # (B,n)
        guard = torch.stack(rows, dim=1)                   # (B,n,n) in [0,1]

        # Map to logits and apply topology mask, then row-softmax
        logits = torch.logit(guard.clamp(eps, 1.0 - eps)) + self.logit_mask  # (B,n,n)
        M_t = torch.softmax(logits / max(float(tau_row), eps), dim=-1)
        return M_t

    @torch.no_grad()
    def extract_all(self, names: List[str], ts: float = 0.5
                    ) -> Dict[Tuple[int, int], List[List[Tuple[str, List[int]]]]]:
        out: Dict[Tuple[int, int], List[List[Tuple[str, List[int]]]]] = {}
        for q in range(self.n):
            for qp in range(self.n):
                if not torch.isfinite(self.logit_mask[q, qp]):
                    continue
                out[(q, qp)] = self._edge(q, qp).extract(names, ts=ts)
        return out


# -----------------------------------------------------------------------------
# NeSy SFA model
# -----------------------------------------------------------------------------
class NeSySFA(nn.Module):
    """
    Forward α through time with time-varying M_t, return acceptance probability.
    """
    def __init__(self, num_states: int, num_dims: int, num_symbols: int, clauses: int,
                 allowed_edges: Optional[List[Tuple[int, int]]] = None,
                 init_gate_logit: float = 0.0):
        super().__init__()
        self.n = int(num_states)
        self.D = int(num_dims)
        self.K = int(num_symbols)

        self.transition = TransitionLayer(
            num_states=self.n,
            edge_class=SymbolDNFEdge,
            edge_kwargs=dict(num_dims=self.D, num_symbols=self.K, R=clauses,
                             init_gate_logit=init_gate_logit),
            allowed_edges=allowed_edges,
        )

        # α0 = one-hot at state 0 for all sequences
        alpha0 = F.one_hot(torch.tensor(0), num_classes=self.n).float().unsqueeze(0)  # (1,n)
        self.register_buffer("alpha0", alpha0)

    def forward(self, sym_seq: torch.Tensor, tau_mask: float = 0.7, tau_row: float = 0.5,
                accept_state: Optional[int] = None) -> torch.Tensor:
        """
        sym_seq : (B,T,D,K) or (B,T,K); returns ŷ : (B,) acceptance prob.
        """
        B = sym_seq.size(0)
        T = sym_seq.size(1)
        if sym_seq.dim() == 3:
            sym_seq = sym_seq.unsqueeze(2)  # (B,T,1,K)

        alpha = self.alpha0.expand(B, -1)   # (B,n)
        for t in range(T):
            M_t = self.transition(sym_seq[:, t, :], tau_mask=tau_mask, tau_row=tau_row)  # (B,n,n)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)                        # (B,n)

        acc = (self.n - 1) if accept_state is None else int(accept_state)
        return alpha[:, acc].clamp(0.0, 1.0)

    @torch.no_grad()
    def forward_with_traces(self, sym_seq: torch.Tensor, tau_mask: float = 0.7, tau_row: float = 0.5
                            ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Return time lists of transition matrices M_t and state beliefs α_t for diagnostics.
        """
        mats, alphas = [], []
        B = sym_seq.size(0)
        T = sym_seq.size(1)
        if sym_seq.dim() == 3:
            sym_seq = sym_seq.unsqueeze(2)

        alpha = self.alpha0.expand(B, -1)
        alphas.append(alpha)
        for t in range(T):
            M_t = self.transition(sym_seq[:, t, :], tau_mask=tau_mask, tau_row=tau_row)
            mats.append(M_t)
            alpha = torch.bmm(alpha.unsqueeze(1), M_t).squeeze(1)
            alphas.append(alpha)
        return mats, alphas


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_bce(model: NeSySFA, loader: DataLoader, device: torch.device,
                 tau_mask: float, tau_row: float, accept_state: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x, y = batch_to_xy(batch, device)
        yhat = model(x, tau_mask=tau_mask, tau_row=tau_row, accept_state=accept_state)
        tot += F.binary_cross_entropy(yhat, y).item() * x.size(0)
        n += x.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def compute_f1(model: NeSySFA, loader: DataLoader, device: torch.device,
               tau_mask: float, tau_row: float, accept_state: int, thr: float = 0.5
               ) -> Tuple[float, float, float]:
    model.eval()
    TP = FP = TN = FN = 0
    for batch in loader:
        x, y = batch_to_xy(batch, device)
        yhat = model(x, tau_mask=tau_mask, tau_row=tau_row, accept_state=accept_state)
        pred = (yhat >= thr).float()
        TP += int(((pred == 1) & (y == 1)).sum().item())
        FP += int(((pred == 1) & (y == 0)).sum().item())
        TN += int(((pred == 0) & (y == 0)).sum().item())
        FN += int(((pred == 0) & (y == 1)).sum().item())

    print("Confusion matrix:")
    print(f"  TP={TP}  FP={FP}")
    print(f"  FN={FN}  TN={TN}")

    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


# -----------------------------------------------------------------------------
# I/O loaders (via your data_structs)
# -----------------------------------------------------------------------------
def _load_dataloaders(train_pt: str, test_pt: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    try:
        from src.asal_nesy.neurasal.data_structs import get_data, get_data_loader
    except Exception as e:
        raise RuntimeError(
            "Cannot import data_structs.get_data/get_data_loader. "
            "Make sure data_structs.py is importable."
        ) from e
    train_data, test_data = get_data(train_pt, test_pt)
    train_loader = get_data_loader(train_data, batch_size, train=True)
    test_loader  = get_data_loader(test_data,  batch_size, train=False)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Symbol-DNF δSFA (clean)")
    ap.add_argument("--train-pt", required=True, type=str)
    ap.add_argument("--test-pt",  required=True, type=str)
    ap.add_argument("--states",   default=4, type=int, help="# SFA states n")
    ap.add_argument("--clauses",  default=1, type=int, help="# DNF clauses per edge R")
    ap.add_argument("--epochs",   default=50, type=int)
    ap.add_argument("--batch-size", default=512, type=int)
    ap.add_argument("--lr",       default=3e-1, type=float)
    ap.add_argument("--tau-row",  default=1.0, type=float, help="row-softmax temperature")  # 0.5
    ap.add_argument("--tau-mask", default=1.0, type=float, help="per-dim symbol-mask temperature") # 0.7
    ap.add_argument("--topology", default="self_fwd", choices=["self_fwd", "full"])
    ap.add_argument("--accepting", default=None, type=int, help="accepting state (default last)")
    ap.add_argument("--device",   default="cuda", type=str)
    ap.add_argument("--save",     default=None, type=str)
    ap.add_argument("--load",     default=None, type=str)
    ap.add_argument("--extract-ts", default=0.05, type=float, help="threshold for extraction (gates & masks)")

    ap.add_argument("--extract-mode", default="cum", choices=["cumulative", "topk"],
                    help="Extraction mode: cumulative mass or 'topk'")
    ap.add_argument("--extract-mass", default=0.90, type=float,
                    help="Target cumulative mass per dimension (cum mode)")
    ap.add_argument("--extract-k", default=2, type=int,
                    help="Top-k per dimension (topk mode)")
    ap.add_argument("--extract-gate", default=0.30, type=float,
                    help="Gate threshold for including a dimension in a clause")
    ap.add_argument("--prune", action="store_true",
                    help="Run pix2rule-style prune/threshold/saturate before extraction")
    ap.add_argument("--prune-eps", default=1e-3, type=float,
                    help="Max allowed ΔvalBCE during pruning steps")
    ap.add_argument("--prune-mag", default=8.0, type=float,
                    help="Saturation magnitude for surviving logits")



    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    # Data
    train_loader, test_loader = _load_dataloaders(args.train_pt, args.test_pt, args.batch_size)  # type: ignore[attr-defined]

    # Peek one batch to infer D,K and set names
    first_batch = next(iter(train_loader))
    x0, _ = batch_to_xy(first_batch, device)
    _, T, D, K = x0.shape
    dim_names = [f"d{i+1}" for i in range(D)]
    names = symbol_names(dim_names, K)

    # Allowed topology
    """
    if args.topology == "self_fwd":
        allowed = [(q, q) for q in range(args.states)] + [(q, q + 1) for q in range(args.states - 1)]
    else:  # full
        allowed = [(q, qp) for q in range(args.states) for qp in range(args.states)]
    """
    from δSFA_utils import build_transitions
    allowed = build_transitions(args.states, allow_backward=False, allow_jumps=False)

    # Model
    model = NeSySFA(
        num_states=args.states,
        num_dims=D,
        num_symbols=K,
        clauses=args.clauses,
        allowed_edges=allowed,
        init_gate_logit=(1.0 if D == 1 else 0.0),  # in univariate, start dim selected moderately
    ).to(device)

    if args.load:
        ckpt = torch.load(args.load, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded weights from {args.load}")

    # Report parameter count
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[PARAMS] Total parameters: {nparams}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    acc_state = (args.states - 1) if args.accepting is None else int(args.accepting)
    print(f"[INFO] SFA n={args.states}  D={D}  K={K}  R={args.clauses}  accepting={acc_state}")
    print(f"[INFO] Allowed edges: {len(allowed)}")


    #--------------------------------------------------------------------------------------------
    def anneal_temp(epoch_idx: int, num_epochs_minus1: int,
                        tau0: float, tau1: float, schedule: str = "cosine") -> float:
        import math
        if schedule == "linear":
            s = epoch_idx / max(1, num_epochs_minus1)
            return tau0 + (tau1 - tau0) * s

        s = epoch_idx / max(1, num_epochs_minus1)
        return tau1 + 0.5 * (tau0 - tau1) * (1 + math.cos(math.pi * s))
    # --------------------------------------------------------------------------------------------
    temp_init, temp_end = 1.0, 0.1

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0

        # temp_mask = anneal_temp(epoch - 1, args.epochs - 1, temp_init, temp_end, 'cosine')
        # temp_row = anneal_temp(epoch - 1, args.epochs - 1, temp_init, temp_end, 'linear')
        # print(temp_mask, temp_row)

        for batch in train_loader:
            x, y = batch_to_xy(batch, device)

            yhat = model(x, tau_mask=args.tau_mask, tau_row=args.tau_row, accept_state=acc_state)
            # yhat = model(x, temp_mask, temp_row, accept_state=acc_state)

            loss = F.binary_cross_entropy(yhat, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = tot / max(n, 1)
        val_loss = evaluate_bce(model, test_loader, device, args.tau_mask, args.tau_row, acc_state)
        print(f"[epoch {epoch:03d}/{args.epochs}] train={train_loss:.4f}  val={val_loss:.4f}")

    # Save
    if args.save:
        torch.save({"model": model.state_dict(),
                    "meta": {"states": args.states, "D": D, "K": K, "R": args.clauses,
                             "allowed": allowed, "accepting": acc_state}},
                   args.save)
        print(f"[INFO] Saved model to {args.save}")

    # Final F1 @ thr=0.5
    f1, prec, rec = compute_f1(model, test_loader, device, args.tau_mask, args.tau_row, acc_state, thr=0.5)
    print("\n[Final @ thr=0.50]  F1={:.4f}  Precision={:.4f}  Recall={:.4f}".format(f1, prec, rec))

    if args.prune:
        print("[INFO] Running prune/threshold/saturate...")
        _ = prune_and_threshold_symbol_dnf(
            model, test_loader, device,
            tau_mask=args.tau_mask, tau_row=args.tau_row,
            accept_state=acc_state,
            eps_drop=args.prune_eps, saturate_mag=args.prune_mag, verbose=True)

    # Build names once (after you infer D,K)
    dim_names = [f"d{i + 1}" for i in range(D)]

    print("\nLearned symbol-DNFs per allowed edge q->q':\n")
    for q in range(args.states):
        for qp in range(args.states):
            # skip disallowed
            if not torch.isfinite(model.transition.logit_mask[q, qp]):
                continue
            edge = model.transition._edge(q, qp)
            if args.extract_mode == "cum":
                clauses = extract_cumulative_mass(edge, dim_names, mass=args.extract_mass, gate_th=args.extract_gate)
            else:
                clauses = extract_topk(edge, dim_names, k=args.extract_k, gate_th=args.extract_gate)

            if not clauses:
                continue

            # pretty print: (d1_is_0 ∨ d1_is_2) ∧ (d2_is_1 ∨ d2_is_9)  OR ...
            def fmt_clause(grouped):
                parts = []
                for dn, syms in grouped:
                    if len(syms) == 1:
                        parts.append(f"{dn}_is_{syms[0]}")
                    else:
                        parts.append("(" + " ∨ ".join(f"{dn}={k}" for k in syms) + ")")
                return " ∧ ".join(parts)

            pretty = "  OR  ".join(fmt_clause(g) for g in clauses)
            print(f"{q}->{qp}:  {pretty}")

    # Extraction
    dnfs = model.transition.extract_all(names, ts=args.extract_ts)
    print("\nLearned symbol-DNFs per allowed edge q->q':\n")
    for (q, qp), clauses in dnfs.items():
        if not clauses:
            continue
        pretty = "  OR  ".join(pretty_symbol_clause_grouped(c) for c in clauses)
        print(f"{q}->{qp}:  {pretty}")


if __name__ == "__main__":
    main()
