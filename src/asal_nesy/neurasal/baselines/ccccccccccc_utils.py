import torch

# --- cumulative-mass extractor (keeps smallest symbol set per dim whose mass >= target) ---
@torch.no_grad()
def extract_cumulative_mass(edge, dim_names, mass: float = 0.90, gate_th: float = 0.30):
    """
    Return a list of grouped clauses for one SymbolDNFEdge:
      clause = [(dim_name, [k1,k2,...]), ...]
    Keeps a dimension in the clause iff gate>=gate_th, and within that dimension
    keeps the smallest set of symbols accumulating at least `mass` probability.
    """
    import torch.nn.functional as F
    gates = torch.sigmoid(edge.gate_logit)     # (R,D)
    masks = F.softmax(edge.mask_logit, dim=-1) # (R,D,K)

    R, D, K = masks.shape
    dnfs = []
    for r in range(R):
        if gates[r].sum() < 1e-9:  # nothing selected
            continue
        clause = []
        for d in range(D):
            if float(gates[r, d]) < gate_th:
                continue
            probs = masks[r, d].detach().cpu()
            order = torch.argsort(probs, descending=True)
            keep, acc = [], 0.0
            for k in order.tolist():
                keep.append(k); acc += float(probs[k])
                if acc >= mass: break
            if keep:
                clause.append((dim_names[d], keep))
        if clause:
            dnfs.append(clause)
    return dnfs


# --- top-k extractor (keep at most k symbols per participating dimension) ---
@torch.no_grad()
def extract_topk(edge, dim_names, k: int = 2, gate_th: float = 0.30):
    """
    Return a list of grouped clauses using top-k symbols per dimension.
    """
    import torch.nn.functional as F
    gates = torch.sigmoid(edge.gate_logit)     # (R,D)
    masks = F.softmax(edge.mask_logit, dim=-1) # (R,D,K)

    R, D, K = masks.shape
    dnfs = []
    for r in range(R):
        if gates[r].sum() < 1e-9:
            continue
        clause = []
        for d in range(D):
            if float(gates[r, d]) < gate_th:
                continue
            probs = masks[r, d].detach().cpu()
            idx = torch.argsort(probs, descending=True)[:k].tolist()
            if idx:
                clause.append((dim_names[d], idx))
        if clause:
            dnfs.append(clause)
    return dnfs


# --- pix2rule-style pruning / thresholding / saturation for symbol-DNF ---
@torch.no_grad()
def evaluate_loader_bce(model, loader, device, tau_mask, tau_row, accept_state):
    import torch.nn.functional as F
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        x, y = batch_to_xy(batch, device)
        yhat = model(x, tau_mask=tau_mask, tau_row=tau_row, accept_state=accept_state)
        tot += F.binary_cross_entropy(yhat, y).item() * x.size(0)
        n += x.size(0)
    return tot / max(n, 1)

@torch.no_grad()
def prune_and_threshold_symbol_dnf(model, val_loader, device,
                                   tau_mask=1.0, tau_row=0.5, accept_state=None,
                                   eps_drop=1e-3, saturate_mag=8.0, verbose=True):
    """
    Post-training: prune → threshold sweep → saturate → (keep best) using a
    validation BCE tolerance eps_drop. This *hardens* masks/gates before extraction.
    """
    base = evaluate_loader_bce(model, val_loader, device, tau_mask, tau_row, accept_state)
    if verbose:
        print(f"[prune] base val BCE={base:.6f}")

    def snapshot():
        return {n: p.clone() for n, p in model.named_parameters()}
    def restore(snap):
        for n, p in model.named_parameters():
            p.copy_(snap[n])

    # 1) Try zeroing small gates
    snap = snapshot()
    for edge in model.transition.edges:
        gl = edge.gate_logit
        thr = 0.10
        edge.gate_logit.copy_(torch.where(gl.abs() < thr, torch.zeros_like(gl), gl))
    new = evaluate_loader_bce(model, val_loader, device, tau_mask, tau_row, accept_state)
    if new - base > eps_drop:
        restore(snap)
    else:
        base = new
        if verbose: print(f"[prune] gates thr ok, val BCE={base:.6f}")

    # 2) Global threshold sweep on |mask_logit|
    with torch.no_grad():
        all_abs = torch.cat([e.mask_logit.abs().flatten() for e in model.transition.edges], 0)
    lo = float(all_abs.quantile(0.10))
    hi = float(all_abs.quantile(0.90))
    best_bce, best_thr, best_snap = base, lo, snapshot()
    for thr in torch.linspace(lo, hi, steps=8):
        snap = snapshot()
        for e in model.transition.edges:
            ml = e.mask_logit
            e.mask_logit.copy_(torch.where(ml.abs() < thr, torch.zeros_like(ml), ml))
        bce = evaluate_loader_bce(model, val_loader, device, tau_mask, tau_row, accept_state)
        if bce - base <= eps_drop and bce <= best_bce:
            best_bce, best_thr, best_snap = bce, float(thr), snapshot()
        restore(snap)  # revert; we’ll reapply best later
    restore(best_snap)
    if verbose:
        print(f"[prune] mask thr={best_thr:.4f}, val BCE={best_bce:.6f}")
    base = best_bce

    # 3) Saturate survivors to ±saturate_mag (near-one-hot masks, 0/1 gates)
    snap = snapshot()
    for e in model.transition.edges:
        ml = e.mask_logit
        nz = (ml != 0)
        e.mask_logit.copy_(torch.where(nz, torch.sign(ml) * saturate_mag, torch.zeros_like(ml)))
        gl = e.gate_logit
        nzg = (gl != 0)
        e.gate_logit.copy_(torch.where(nzg, torch.sign(gl) * saturate_mag, torch.zeros_like(gl)))
    new = evaluate_loader_bce(model, val_loader, device, tau_mask, tau_row, accept_state)
    if new - base > eps_drop:
        restore(snap)
    else:
        base = new
        if verbose: print(f"[prune] saturate ok, val BCE={base:.6f}")

    return base



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