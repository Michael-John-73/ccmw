"""Exact GPU re-implementation of the upstream `multiwm_dbscan` call (sklearn DBSCAN) for binary bits.

Why it is exact: points are 0/1 vectors, so Euclidean distance <= eps=1 <=> Hamming distance <= 1.
Identical bit patterns share one neighbourhood, so DBSCAN can run on unique patterns weighted by counts:
- core: (sum of counts of patterns within Hamming 1, self included) >= min_samples (sklearn counts self);
- clusters: connected components of core patterns, numbered in increasing order of the smallest pixel
  index they contain (sklearn starts a new cluster at the first unlabelled core point in index order);
- border points: the smallest cluster number among their core neighbours (that cluster's depth-first
  expansion finishes before any later cluster starts); otherwise noise (-1);
- centroid: per-cluster mean of bits > 0.5, as upstream.
Equality with the upstream function is checked on real WAM outputs before this path is used.
"""
import numpy as np
import torch

_POW = None


def _pack(bits):
    global _POW
    if _POW is None or _POW.device != bits.device:
        _POW = (2 ** torch.arange(bits.shape[1], device=bits.device, dtype=torch.int64))
    return (bits.long() * _POW).sum(1)


def dbscan_bits(logits, min_samples=1000, eps=1.0):
    """logits: (33, H, W) tensor (any device). Returns ({label: uint8[32]}, labels int32 (H, W))."""
    if eps != 1.0:
        raise ValueError("Exact equivalence is derived for eps = 1 on binary bits only")
    k, h, w = logits.shape[0] - 1, logits.shape[1], logits.shape[2]
    # Mask exactly as upstream: sigmoid evaluated on CPU (GPU sigmoid can round differently at 0.5).
    mask = (torch.sigmoid(logits[0].detach().cpu()).reshape(-1) > .5).to(logits.device)
    full = torch.full((h * w,), -1, dtype=torch.int64, device=logits.device)
    if not bool(mask.any()):
        return {}, full.reshape(h, w).cpu().numpy().astype(np.int32)
    idx = torch.nonzero(mask, as_tuple=False)[:, 0]                       # pixel order = upstream valid order
    bits = (logits[1:].reshape(k, -1).t()[idx] > 0)                        # (n, 32) bool, upstream threshold 0
    codes = _pack(bits)
    uniq, inverse, counts = torch.unique(codes, sorted=True, return_inverse=True, return_counts=True)
    u = uniq.shape[0]
    order = torch.arange(idx.shape[0], device=logits.device)
    first = torch.full((u,), idx.shape[0], dtype=torch.int64, device=logits.device).scatter_reduce(
        0, inverse, order, reduce="amin")
    flips = uniq[:, None] ^ _POW[None, :]                                  # (u, 32) Hamming-1 neighbours
    pos = torch.searchsorted(uniq, flips).clamp_max(u - 1)
    found = uniq[pos] == flips
    nbr = torch.where(found, pos, torch.full_like(pos, -1))                # (u, 32), -1 = absent
    ncount = counts + torch.where(found, counts[pos], torch.zeros_like(pos)).sum(1)
    core = ncount >= min_samples
    big = idx.shape[0] + 1
    comp = torch.where(core, first, torch.full_like(first, big))
    valid_edge = found & core[pos] & core[:, None]
    while True:                                                            # min-label propagation on core graph
        neigh = torch.where(valid_edge, comp[pos], torch.full_like(pos, big)).min(1).values
        new = torch.where(core, torch.minimum(comp, neigh), comp)
        if torch.equal(new, comp):
            break
        comp = new
    roots = torch.unique(comp[core])                                       # sorted = cluster numbering
    label = torch.full((u,), -1, dtype=torch.int64, device=logits.device)
    if roots.numel():
        label[core] = torch.searchsorted(roots, comp[core])
        border_cand = torch.where(found & core[pos], label[pos], torch.full_like(pos, big))
        best = border_cand.min(1).values
        border = (~core) & (best < big)
        label[border] = best[border]
    pix = label[inverse]
    full[idx] = pix
    centers = {}
    if roots.numel():
        n_clusters = roots.numel()
        keep = pix >= 0
        sums = torch.zeros((n_clusters, k), device=logits.device).index_add_(0, pix[keep], bits[keep].float())
        sizes = torch.zeros(n_clusters, device=logits.device).index_add_(0, pix[keep], torch.ones_like(pix[keep], dtype=torch.float))
        cent = (sums / sizes[:, None]) > .5
        centers = {int(c): cent[c].to(torch.uint8).cpu().numpy() for c in range(n_clusters)}
    return centers, full.reshape(h, w).cpu().numpy().astype(np.int32)
