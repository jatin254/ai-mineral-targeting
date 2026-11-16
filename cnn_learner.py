"""
cnn_learner.py - the CNN contender for train_v1.py (--learner cnn).

A small fully convolutional network: it looks at a 1.7 km x 1.7 km window (17 x 17 cells) of every input layer
around a cell and scores that cell. Trained on windows around the gold labels and the background sample (rotated
and mirrored at random, since gold has no preferred direction here), then run over the whole grid in tiles.
Inputs are scaled by median / interquartile range over the model's ground; blanks become 0 (the median).
Needs PyTorch: pip install torch
"""
from __future__ import annotations

import os

import numpy as np

try:
    import torch
    from torch import nn
except (ImportError, OSError) as e:  # pragma: no cover
    raise SystemExit("The CNN contender needs PyTorch, which did not load here: "
                     f"{type(e).__name__}: {str(e)[:120]}\n"
                     "On Windows with conda, use the conda-forge build:  pip uninstall -y torch  then  "
                     "conda install -c conda-forge pytorch") from e

HALO = 8          # receptive field 17 cells = 1 + 2 + 2 + 4 + 8
TILE = 512


def build_stack(X: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple, dom: np.ndarray) -> np.ndarray:
    """(channels, H + 2*HALO, W + 2*HALO) float32: robust-scaled layers, 0 outside the mask and where blank."""
    C = X.shape[1]
    out = np.zeros((C, shape[0] + 2 * HALO, shape[1] + 2 * HALO), dtype="float32")
    for j in range(C):
        v = X[:, j].astype("float64")
        ref = v[dom & np.isfinite(v)]
        if len(ref) == 0:
            continue
        med = np.median(ref)
        iqr = np.subtract(*np.percentile(ref, [75, 25]))
        z = np.clip((v - med) / (iqr if iqr > 0 else (np.std(ref) or 1.0)), -5, 5)
        out[j, rows + HALO, cols + HALO] = np.where(np.isfinite(z), z, 0.0).astype("float32")
    return out


def _net(C: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(C, 16, 3), nn.ReLU(),
        nn.Conv2d(16, 16, 3), nn.ReLU(),
        nn.Conv2d(16, 32, 3, dilation=2), nn.ReLU(),
        nn.Conv2d(32, 32, 3, dilation=4), nn.ReLU(),
        nn.Conv2d(32, 1, 1))


def _patches(stack: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    k = np.arange(2 * HALO + 1)
    rr, cc = r[:, None] + k, c[:, None] + k
    p = stack[:, rr[:, :, None], cc[:, None, :]]          # (C, N, 17, 17)
    return np.ascontiguousarray(p.transpose(1, 0, 2, 3))


def train_score_map(stack: np.ndarray, pos_rc: tuple, pos_w: np.ndarray, neg_rc: tuple, neg_w: np.ndarray,
                    seed: int, epochs: int = 15, bg_max: int = 4000, run_mask: np.ndarray | None = None) -> np.ndarray:
    """Train one network; return its score (0-1) for every grid cell, shape (H, W)."""
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng = np.random.default_rng(seed)
    nr, nc = neg_rc
    if len(nr) > bg_max:                                   # background subsample (the CNN is slower than trees)
        keep = rng.choice(len(nr), size=bg_max, replace=False)
        nr, nc, neg_w = nr[keep], nc[keep], neg_w[keep]
    pr, pc = pos_rc
    Xp = np.concatenate([_patches(stack, pr, pc), _patches(stack, nr, nc)])
    y = np.r_[np.ones(len(pr)), np.zeros(len(nr))].astype("float32")
    w = np.r_[pos_w * (neg_w.sum() / max(pos_w.sum(), 1e-9)), neg_w].astype("float32")   # classes weigh the same
    w = w / w.mean()
    Xt, yt, wt = torch.from_numpy(Xp), torch.from_numpy(y), torch.from_numpy(w)
    net = _net(stack.shape[0])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    n = len(yt)
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(n))
        net.train()
        for s in range(0, n, 256):
            ix = order[s:s + 256]
            xb = Xt[ix]
            k = int(rng.integers(4))
            if k:
                xb = torch.rot90(xb, k, dims=(2, 3))
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=(3,))
            out = net(xb).reshape(-1)
            loss = (lossf(out, yt[ix]) * wt[ix]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    # whole grid, tile by tile (tiles without model ground are skipped)
    net.eval()
    H, W = stack.shape[1] - 2 * HALO, stack.shape[2] - 2 * HALO
    score = np.zeros((H, W), dtype="float32")
    with torch.no_grad():
        for r0 in range(0, H, TILE):
            for c0 in range(0, W, TILE):
                r1, c1 = min(H, r0 + TILE), min(W, c0 + TILE)
                if run_mask is not None and not run_mask[r0:r1, c0:c1].any():
                    continue
                x = torch.from_numpy(np.ascontiguousarray(stack[None, :, r0:r1 + 2 * HALO, c0:c1 + 2 * HALO]))
                score[r0:r1, c0:c1] = torch.sigmoid(net(x))[0, 0].numpy()
    return score
