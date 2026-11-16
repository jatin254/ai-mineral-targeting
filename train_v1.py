#!/usr/bin/env python3
"""
train_v1.py - first gold prospectivity model with honest, region-held-out scoring.

Run from the repository folder, after build_geochem_grid.py. The V1 comparison is three runs:
    python train_v1.py data/processed --name pathfinders --only gc_au_index,gc_au,gc_as,gc_sb,gc_bi --monotonic
    python train_v1.py data/processed --name pathf_geophys --only gc_au_index,gc_au,gc_as,gc_sb,gc_bi,rad_,grav_,mag_ --monotonic
    python train_v1.py data/processed --name everything --monotonic
Production run (bagged, see below):
    python train_v1.py data/processed --name pg_ens --only gc_au_index,gc_au,gc_as,gc_sb,gc_bi,rad_,grav_,mag_ --monotonic --bags 8

Reads   data/processed/stack/*.tif (features), mask_model.tif, ka_ap_v1.gpkg (gold_labels)
Writes  data/processed/models/<name>/
          prospectivity_pct.tif    percentile rank inside the model mask (100 = most prospective)
          prospectivity.png        map with gold labels
          cv_regions.csv           per held-out region: capture in top 1/5/10/20% for model and references
          group_importance.csv     drop in held-out AUC when each feature group is shuffled
          run.json                 settings, features, label counts, summary metrics
          bags/pct_bagNN.tif       with --bags > 1: each bag's own percentile map (make_targets --stability bags)

Method (presence-background, the usual way to handle positive-only labels):
  * positives = gold label cells (occurrences weight 1, merged old workings 0.25)
  * domain (default "belt") = ground within 1 km of greenstone (--belt-km to widen). The model learns and ranks only there;
    ground outside is ranked below it by distance to the belts. "--domain all" learns on everything.
  * background = random cells of the domain, at least 500 m from any positive
  * model = histogram gradient boosting (handles missing values and the lithology category);
    --monotonic forces gold pathfinders (Au, As, Sb, Bi, Ag, Hg, W, index) to only raise the score
  * scoring = leave-one-region-out: positives are grouped into spatial regions; each region is held
    out in turn, the model is trained on the rest, and all cells of the held-out region are ranked
    (ties broken at random, so "top 10%" is exactly 10% of the ground).
  * --bags N: N models, each with its own random background sample, averaged (bagging for
    positive-unlabeled data; Mordelet & Vert 2014, Pattern Recognition Letters 37:201-209). With ~80
    positives the background draw alone moved held-out capture a lot (V1 seed runs: within-greenstone
    top 10% from 17% to 55% on identical regions), so single runs are not reliable; the average is.
    Hold-out regions are fixed by --region-seed, so runs with different --seed are comparable.
  * terrain (needs data/processed/terrain from build_terrain_layers.py; not model features):
    - always reports capture WITHIN THE SAME TERRAIN: each held-out label is ranked only against belt
      ground in its own relief-quintile x scrub-share bin. A model that only learned "exposed rock"
      scores ~10% here; real geological skill shows up as > 10%.
    - --match-terrain S (0-1) samples background so its terrain matches the labels' (S=1 fully,
      0.5 halfway), so terrain alone cannot separate gold from background (target-group background,
      Phillips et al. 2009, Ecological Applications 19:181-197).
  * blend with the Au pathfinder index: CV always also scores a 50/50 rank average of the model and
    the index (the index is not trained, so this does not leak). --blend-au W makes the final map
    that blend (W = weight of the index). Where there is no gold geochemistry the model's rank is used.
  * --terrain-correct rad_ : replaces each radiometric layer by what terrain and land cover do not
    explain (residual of a regression on relief, scrub, crop, tree, built, water over belt ground),
    so radiometrics describe the rock rather than how exposed it is.
  * references, no training: (1) greenstone only; (2) greenstone, then the gold pathfinder index.
    The model earns its place only if it beats (2), and the key number is how much held-out gold it
    puts in the top 10% of greenstone (random = 10%).
Finding from the first runs (geology, structure, geochemistry): lithology and structure rules do not
transfer between belts and lower within-belt skill; the pathfinder geochemistry does transfer.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import Affine, rowcol
    from scipy import ndimage
    from scipy.stats import rankdata
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

CATEGORICAL = {"geo_litho_class"}
GROUPS = [("magnetics", "mag_"), ("gravity", "grav_"), ("radiometrics", "rad_"),
          ("lithology", "geo_"), ("structure", "struct_"), ("geochemistry", "gc_"),
          ("shear zones (V2)", "v2s_"), ("soil / stream geochemistry (V2)", "v2g_")]
CAPTURE_AT = (1, 5, 10, 20)
PATHFINDERS = {"gc_au_index", "gc_au", "gc_as", "gc_sb", "gc_bi", "gc_ag", "gc_hg", "gc_w",   # higher = more prospective
               "v2g_soil_au", "v2g_soil_as", "v2g_soil_sb", "v2g_soil_bi", "v2g_soil_index",
               "v2g_ss_au", "v2g_ss_as", "v2g_ss_sb", "v2g_ss_bi", "v2g_ss_index"}
BG_EXCLUSION_M = 500.0
BELT_BUFFER_M = 1000.0   # --domain belt: model ground within 1 km of a greenstone belt
CHUNK = 500_000
MODEL_PARAMS = dict(learning_rate=0.05, max_iter=250, max_depth=4, max_leaf_nodes=15,
                    min_samples_leaf=50, l2_regularization=1.0)


# ----------------------------------------------------------------------------- helpers

def load_grid(proc: Path):
    spec = json.loads((proc / "grid_spec.json").read_text())
    cell = float(spec["cell_m"])
    T = Affine(cell, 0, spec["xmin"], 0, -cell, spec["ymax"])
    return spec, cell, T, (int(spec["height"]), int(spec["width"]))


def group_of(name: str) -> str:
    for g, p in GROUPS:
        if name.startswith(p):
            return g
    return "other"


def make_model(cat_mask, seed, monotonic=None):
    return HistGradientBoostingClassifier(categorical_features=cat_mask, monotonic_cst=monotonic,
                                          random_state=seed, **MODEL_PARAMS)


def predict(model, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X), dtype="float32")
    for s in range(0, len(X), CHUNK):
        out[s:s + CHUNK] = model.predict_proba(X[s:s + CHUNK])[:, 1]
    return out


def tie_broken_rank(scores: np.ndarray, rng) -> np.ndarray:
    """Rank 0..n-1 by score with random tie-breaking, so 'top 10%' is exactly 10% of the ground
    even when many cells share a score (trees and grid distances produce lots of ties)."""
    order = np.lexsort((rng.random(len(scores)), scores))
    rank = np.empty(len(scores), dtype="int64")
    rank[order] = np.arange(len(scores))
    return rank


def capture(rank: np.ndarray, pos_idx: np.ndarray, pos_w: np.ndarray, pct: float) -> float:
    """Weighted share of positives whose cell is in the top `pct`% of the ranked cells."""
    if len(pos_idx) == 0 or pos_w.sum() == 0:
        return np.nan
    cutoff = len(rank) * (1 - pct / 100.0)
    return float(pos_w[rank[pos_idx] >= cutoff].sum() / pos_w.sum())


def balanced_weights(y: np.ndarray, w_pos: np.ndarray) -> np.ndarray:
    w = np.ones(len(y), dtype="float64")
    w[y == 1] = w_pos
    w[y == 1] *= (y == 0).sum() / max(w_pos.sum(), 1e-9)   # positive weight total = background count
    return w


def gsi_folds(xy: np.ndarray, d_m: float) -> np.ndarray:
    """Two folds of GSI sites: single-linkage clusters (sites closer than d_m), largest clusters first,
    each into the fold with fewer sites so far. Same sites -> same folds on every run."""
    n = len(xy)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        d = np.hypot(xy[i + 1:, 0] - xy[i, 0], xy[i + 1:, 1] - xy[i, 1])
        for j in np.nonzero(d <= d_m)[0] + i + 1:
            parent[find(i)] = find(j)
    root = np.array([find(i) for i in range(n)])
    sizes = sorted(((int((root == r).sum()), int(r)) for r in np.unique(root)), reverse=True)
    fold, tot = np.zeros(n, dtype=int), [0, 0]
    for sz, r in sizes:
        f = int(tot[1] < tot[0])
        fold[root == r] = f
        tot[f] += sz
    return fold


def weighted_auc(y, s, w):
    try:
        return float(roc_auc_score(y, s, sample_weight=w))
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the V1 gold prospectivity model.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("--name", default="full", help="run name (output folder under models/)")
    ap.add_argument("--exclude", default="", help="comma-separated feature prefixes to drop, e.g. gc_ or mag_,grav_")
    ap.add_argument("--only", default="", help="comma-separated feature names/prefixes to keep (everything else dropped)")
    ap.add_argument("--monotonic", action="store_true",
                    help="force gold pathfinders (Au, As, Sb, Bi, Ag, Hg, W, index) to only raise prospectivity")
    ap.add_argument("--regions", type=int, default=5, help="number of spatial hold-out regions")
    ap.add_argument("--background", type=int, default=50_000, help="background cells sampled")
    ap.add_argument("--domain", choices=["belt", "all"], default="belt",
                    help="belt: learn and rank only within 1 km of greenstone, ground outside ranked by distance "
                         "to the belts (default). all: learn on the whole mask")
    ap.add_argument("--belt-km", type=float, default=BELT_BUFFER_M / 1000,
                    help="belt domain = ground within this distance of greenstone (default 1 km); wider reaches "
                         "gold at granite contacts just off the mapped belts")
    ap.add_argument("--learner", choices=["hgb", "cnn"], default="hgb",
                    help="hgb: gradient-boosted trees on each cell's values (default). cnn: a small convolutional "
                         "network on the 1.7 km window round each cell (needs: pip install torch)")
    ap.add_argument("--cnn-epochs", type=int, default=15)
    ap.add_argument("--cnn-bg", type=int, default=4000, help="background cells per CNN bag")
    ap.add_argument("--gsi-sites", default="",
                    help="gsi_label_sites.csv: GSI 'positive' sites (>= 0.5 g/t rock) join the training labels and "
                         "barren sites become known negatives. Cross-validation still scores only the original labels")
    ap.add_argument("--gsi-pos-weight", type=float, default=1.0, help="label weight of a GSI positive site")
    ap.add_argument("--gsi-neg-weight", type=float, default=50.0,
                    help="each barren GSI site counts as this many background cells")
    ap.add_argument("--gsi-holdout", choices=["none", "A", "B"], default="none",
                    help="leave out GSI fold A or B (clusters of sites) to test on it afterwards with gsi_eval.py")
    ap.add_argument("--gsi-cluster-km", type=float, default=5.0, help="GSI sites closer than this share a fold")
    ap.add_argument("--seed", type=int, default=42, help="background sampling / model seed")
    ap.add_argument("--bags", type=int, default=1, help="models averaged, each on its own background sample")
    ap.add_argument("--region-seed", type=int, default=42, help="seed for the hold-out regions (keep fixed)")
    ap.add_argument("--terrain-dir", default="terrain", help="folder under processed/ with relief.tif and scrub.tif")
    ap.add_argument("--blend-au", type=float, default=0.0,
                    help="final map = rank blend of model and Au pathfinder index with this index weight (0-1)")
    ap.add_argument("--terrain-correct", default="", help="feature prefixes to correct for terrain, e.g. rad_")
    ap.add_argument("--match-terrain", type=float, default=0.0,
                    help="0 = random background; 1 = background terrain matched to the labels; 0.5 = halfway")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    proc = Path(args.processed)
    out = proc / "models" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    spec, cell, T, shape = load_grid(proc)

    # ---- mask + features
    with rasterio.open(proc / "mask_model.tif") as src:
        mask = src.read(1) == 1
    rows, cols = np.nonzero(mask)
    n = len(rows)
    excl = [p.strip() for p in args.exclude.split(",") if p.strip()]
    only = [p.strip() for p in args.only.split(",") if p.strip()]
    paths = sorted(p for p in (proc / "stack").glob("*.tif")
                   if not any(p.stem.startswith(e) for e in excl)
                   and (not only or any(p.stem == o or p.stem.startswith(o) for o in only)))
    names = [p.stem for p in paths]
    X = np.empty((n, len(paths)), dtype="float32")
    for j, p in enumerate(paths):
        with rasterio.open(p) as src:
            X[:, j] = src.read(1)[rows, cols]
    cat_mask = [nm in CATEGORICAL for nm in names]
    mono = [1 if (args.monotonic and nm in PATHFINDERS) else 0 for nm in names] if args.monotonic else None
    print(f"Loaded {len(names)} features for {n:,} cells  [{time.time() - t0:.0f}s]", flush=True)

    # baseline score: inside / near greenstone (independent of --exclude)
    with rasterio.open(proc / "stack" / "geo_greenstone_sdist.tif") as src:
        base_score = -src.read(1)[rows, cols]
    base_score = np.where(np.isfinite(base_score), base_score, np.nanmin(base_score))
    with rasterio.open(proc / "stack" / "geo_in_greenstone.tif") as src:
        in_green = src.read(1)[rows, cols] == 1
    belt_m = args.belt_km * 1000.0
    dom = (-base_score <= belt_m) if args.domain == "belt" else np.ones(n, dtype=bool)
    near_belt = -base_score <= belt_m
    with rasterio.open(proc / "stack" / "gc_au_index.tif") as src:
        au_idx = src.read(1)[rows, cols].astype("float64")
    # reference 2 (no training): belts first, ranked inside by the gold pathfinder index (missing = lowest)
    ref2 = np.where(near_belt, 1e9 + np.where(np.isfinite(au_idx), au_idx, -50.0), base_score)

    def combine(model_score: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Belt domain: every belt cell outranks every non-belt cell; inside, the model decides;
        outside, nearer the belts ranks higher."""
        if args.domain == "all":
            return model_score.astype("float64")
        return np.where(dom[cells], 1e9 + model_score.astype("float64"), base_score[cells].astype("float64"))

    # ---- positives
    lab = gpd.read_file(proc / args.gpkg, layer="gold_labels")
    gsi_info, neg_xy = {}, None
    if args.gsi_sites:
        gs = pd.read_csv(args.gsi_sites)
        gs = gs[gs["label"].isin(["positive", "anomalous", "barren_strong", "barren"])].reset_index(drop=True)
        tr_ = Transformer.from_crs("EPSG:4326", lab.crs, always_xy=True)
        gx, gy = (np.asarray(v) for v in tr_.transform(gs["lon"].to_numpy(), gs["lat"].to_numpy()))
        fold = gsi_folds(np.c_[gx, gy], args.gsi_cluster_km * 1000)
        hold = {"none": None, "A": 0, "B": 1}[args.gsi_holdout]
        use = np.ones(len(gs), dtype=bool) if hold is None else fold != hold
        if hold is not None:
            gs.loc[~use].assign(fold="AB"[hold]).to_csv(out / "gsi_holdout_sites.csv", index=False)
        lx, ly = lab.geometry.x.to_numpy(), lab.geometry.y.to_numpy()
        dmin = np.array([np.min(np.hypot(lx - x, ly - y)) for x, y in zip(gx, gy)])
        posm = use & (gs["label"] == "positive").to_numpy() & (dmin > 250)   # within 250 m: already a label
        add = gpd.GeoDataFrame({"label_id": [f"GSI{i + 1:03d}" for i in np.nonzero(posm)[0]], "label_class": "gsi_positive",
                                "weight": args.gsi_pos_weight}, geometry=gpd.points_from_xy(gx[posm], gy[posm]), crs=lab.crs)
        lab = gpd.GeoDataFrame(pd.concat([lab, add], ignore_index=True), crs=lab.crs)
        negm = use & gs["label"].isin(["barren_strong", "barren"]).to_numpy()
        neg_xy = (gx[negm], gy[negm])
        gsi_info = {"gsi_sites": str(args.gsi_sites), "gsi_holdout": args.gsi_holdout, "gsi_positives_added": int(posm.sum()),
                    "gsi_barren_sites": int(negm.sum()), "gsi_neg_weight": args.gsi_neg_weight,
                    "gsi_held_out_sites": int((~use).sum())}
        print(f"  GSI: {int(posm.sum())} positive sites added as labels, {int(negm.sum())} barren sites as negatives"
              + (f", fold {args.gsi_holdout} held out ({int((~use).sum())} sites)" if hold is not None else ""))
    lr, lc_ = rowcol(T, lab.geometry.x.values, lab.geometry.y.values)
    lr, lc_ = np.asarray(lr), np.asarray(lc_)
    inb = (lr >= 0) & (lr < shape[0]) & (lc_ >= 0) & (lc_ < shape[1])
    index_grid = np.full(shape, -1, dtype="int64")
    index_grid[rows, cols] = np.arange(n)
    cell_idx = np.full(len(lab), -1)
    cell_idx[inb] = index_grid[lr[inb], lc_[inb]]
    lab["cell"] = cell_idx
    inside = lab[lab.cell >= 0]
    print(f"Gold labels inside the model mask: {len(inside)} of {len(lab)} "
          f"(occurrences {int((inside.label_class == 'occurrence').sum())} of {int((lab.label_class == 'occurrence').sum())})")
    missing = lab.loc[lab.cell < 0, "label_id"].tolist()
    if missing:
        print(f"  outside mask, not used: {', '.join(missing)}")
    pos = inside.groupby("cell").agg(w=("weight", "max"), occ=("label_class", lambda s: (s == "occurrence").any()),
                                     gsi=("label_class", lambda s: (s == "gsi_positive").all())).reset_index()
    if args.domain == "belt":
        keep = dom[pos["cell"].to_numpy()]
        print(f"  belt domain ({dom.mean():.1%} of mask): {int(keep.sum())} positive cells used, "
              f"{int((~keep).sum())} more than {belt_m:.0f} m outside greenstone left out")
        pos = pos[keep].reset_index(drop=True)
    pos_cell = pos["cell"].to_numpy()
    pos_w = pos["w"].to_numpy(dtype="float64")
    is_gsi = pos["gsi"].to_numpy(dtype=bool)          # GSI positives: trained on, never scored in cross-validation

    # ---- regions: k-means on positive locations, every cell joins its nearest centre
    pxy = np.c_[cols[pos_cell], rows[pos_cell]].astype("float64")
    km = KMeans(n_clusters=args.regions, n_init=20, random_state=args.region_seed).fit(pxy[~is_gsi], sample_weight=pos_w[~is_gsi])
    centres = km.cluster_centers_
    region = np.empty(n, dtype="int16")
    for s in range(0, n, CHUNK):
        d = ((np.c_[cols[s:s + CHUNK], rows[s:s + CHUNK]][:, None, :] - centres[None]) ** 2).sum(-1)
        region[s:s + CHUNK] = d.argmin(1)
    pos_region = region[pos_cell]

    # ---- terrain bins (relief quintile on belt ground x scrub-share third), for metrics / matching
    tdir = proc / args.terrain_dir
    tbin, relief_c, scrub_c = None, None, None
    if (tdir / "relief.tif").exists() and (tdir / "scrub.tif").exists():
        with rasterio.open(tdir / "relief.tif") as src:
            relief_c = src.read(1)[rows, cols].astype("float64")
        with rasterio.open(tdir / "scrub.tif") as src:
            scrub_c = src.read(1)[rows, cols].astype("float64")
        tok = np.isfinite(relief_c) & np.isfinite(scrub_c)
        edges = np.quantile(relief_c[tok & dom], [0.2, 0.4, 0.6, 0.8])
        tbin = np.full(n, -1, dtype="int64")
        tbin[tok] = np.searchsorted(edges, relief_c[tok], side="right") * 3 + np.minimum((scrub_c[tok] * 3).astype(int), 2)
    elif args.match_terrain > 0:
        sys.exit("--match-terrain needs data/processed/terrain: run build_terrain_layers.py first")
    N_TBIN = 15

    # ---- terrain-corrected features (residuals of a terrain/land-cover regression on belt ground)
    corr_pref = [p_.strip() for p_ in args.terrain_correct.split(",") if p_.strip()]
    terrain_r2 = {}
    if corr_pref:
        cov_names = [k for k in ("relief", "scrub", "crop", "tree", "built", "water") if (tdir / f"{k}.tif").exists()]
        if tbin is None or not cov_names:
            sys.exit("--terrain-correct needs data/processed/terrain: run build_terrain_layers.py first")
        C = np.empty((n, len(cov_names)), dtype="float32")
        for j, k in enumerate(cov_names):
            with rasterio.open(tdir / f"{k}.tif") as src:
                C[:, j] = src.read(1)[rows, cols]
        cok = np.isfinite(C).all(1)
        fit_pool = np.nonzero(cok & dom)[0]
        fit_idx = np.random.default_rng(args.seed).choice(fit_pool, size=min(300_000, len(fit_pool)), replace=False)
        for j, nm in enumerate(names):
            if not any(nm.startswith(p_) for p_ in corr_pref):
                continue
            y = X[fit_idx, j].astype("float64")
            ok = np.isfinite(y)
            if ok.sum() < 1000:
                continue
            reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                                random_state=0).fit(C[fit_idx][ok], y[ok])
            pred = np.full(n, np.nan, dtype="float32")
            idx = np.nonzero(cok)[0]
            for s0 in range(0, len(idx), CHUNK):
                pred[idx[s0:s0 + CHUNK]] = reg.predict(C[idx[s0:s0 + CHUNK]])
            yf = y[ok]
            terrain_r2[nm] = float(1 - np.mean((yf - pred[fit_idx][ok]) ** 2) / max(np.var(yf), 1e-12))
            X[:, j] = np.where(cok & np.isfinite(X[:, j]), X[:, j] - pred, np.nan)
        print("  terrain-corrected: " + ", ".join(f"{k} (terrain explained {v:.0%})" for k, v in terrain_r2.items()))

    # ---- background samples (>= 500 m from any positive), one per bag
    pres = np.zeros(shape, dtype=bool)
    pres[rows[pos_cell], cols[pos_cell]] = True
    far = ndimage.distance_transform_edt(~pres)[rows, cols] * cell >= BG_EXCLUSION_M
    cand = np.nonzero(far & dom)[0]
    n_bags = max(1, args.bags)
    bag_seeds = [args.seed + 7919 * b for b in range(n_bags)]
    p_cand = None
    if args.match_terrain > 0:
        pb, cb = tbin[pos_cell], tbin[cand]
        wl = np.bincount(pb[pb >= 0], weights=pos_w[pb >= 0], minlength=N_TBIN)
        cl = np.bincount(cb[cb >= 0], minlength=N_TBIN).astype("float64")
        share_l = (wl + 0.5) / (wl.sum() + 0.5 * N_TBIN)      # smoothed: no bin is impossible
        share_c = (cl + 1.0) / (cl.sum() + N_TBIN)
        wbin = (share_l / share_c) ** args.match_terrain
        p_cand = np.where(cb >= 0, wbin[np.maximum(cb, 0)], np.median(wbin))
        p_cand = p_cand / p_cand.sum()
    bg_cells = [np.random.default_rng(sd).choice(cand, size=min(args.background, len(cand)), replace=False, p=p_cand)
                for sd in bag_seeds]
    if tbin is not None:
        def mix(cells_):
            return (float(np.nanmedian(relief_c[cells_])), float(np.nanmean(scrub_c[cells_] >= 0.5)))
        (rl, sl), (rb, sb), (rc, sc) = mix(pos_cell), mix(bg_cells[0]), mix(cand)
        print(f"  terrain (median relief m / share mostly scrub): labels {rl:.0f} / {sl:.0%}, "
              f"background {rb:.0f} / {sb:.0%}, all belt ground {rc:.0f} / {sc:.0%}"
              + (f"  [matched, strength {args.match_terrain:g}]" if args.match_terrain > 0 else ""))
    bg_regions = [region[b] for b in bg_cells]
    neg_cell = np.array([], dtype="int64")
    if neg_xy is not None and len(neg_xy[0]):
        nr, nc = (np.asarray(v) for v in rowcol(T, neg_xy[0], neg_xy[1]))
        ok = (nr >= 0) & (nr < shape[0]) & (nc >= 0) & (nc < shape[1])
        idx = index_grid[nr[ok], nc[ok]]
        idx = idx[idx >= 0]
        neg_cell = np.unique(idx[dom[idx] & far[idx]])     # on the model's ground and >= 500 m from any label
        print(f"  GSI barren sites used as negatives: {len(neg_cell)} cells (weight {args.gsi_neg_weight:g} each)")
        gsi_info["gsi_negative_cells"] = int(len(neg_cell))
    neg_region = region[neg_cell]
    if n_bags > 1:
        print(f"  bagging: {n_bags} models, each with its own {len(bg_cells[0]):,}-cell background sample")

    cnn_stack = None
    if args.learner == "cnn":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cnn_learner import build_stack, train_score_map
        cnn_stack = build_stack(X, rows, cols, shape, dom)
        dom_grid = np.zeros(shape, dtype=bool)
        dom_grid[rows[dom], cols[dom]] = True
        print(f"  CNN: {cnn_stack.shape[0]} input layers, 17 x 17-cell window, {args.cnn_epochs} epochs, "
              f"{args.cnn_bg:,} background cells per bag (monotonic constraints do not apply)")

    def fit_bags(pos_sel, bg_sels, neg_sel=None):
        models = []
        ng = neg_cell if neg_sel is None else neg_cell[neg_sel]
        if args.learner == "cnn":
            pc = pos_cell[pos_sel]
            for bgc, sd in zip(bg_sels, bag_seeds):
                nc_ = np.r_[bgc, ng]
                nw = np.r_[np.ones(len(bgc)), np.full(len(ng), args.gsi_neg_weight)]
                models.append(train_score_map(cnn_stack, (rows[pc], cols[pc]), pos_w[pos_sel], (rows[nc_], cols[nc_]), nw,
                                              sd, epochs=args.cnn_epochs, bg_max=args.cnn_bg, run_mask=dom_grid))
            return models
        for b, (bgc, sd) in enumerate(zip(bg_sels, bag_seeds)):
            Xtr = np.vstack([X[pos_cell[pos_sel]], X[bgc], X[ng]])
            ytr = np.r_[np.ones(int(pos_sel.sum())), np.zeros(len(bgc) + len(ng))]
            wtr = balanced_weights(ytr, pos_w[pos_sel])
            if len(ng):
                wtr[-len(ng):] *= args.gsi_neg_weight
            models.append(make_model(cat_mask, sd, mono).fit(Xtr, ytr, sample_weight=wtr))
        return models

    def predict_dom(models, cells):
        """Per-bag model scores for `cells`; outside the belt domain the model is not used (score 0)."""
        out = np.zeros((len(models), len(cells)), dtype="float32")
        dm = dom[cells]
        if dm.any():
            if args.learner == "cnn":   # a CNN model is its score map
                cd = cells[dm]
                for b, m in enumerate(models):
                    out[b, dm] = m[rows[cd], cols[cd]]
                return out
            Xc = X[cells[dm]]
            for b, m in enumerate(models):
                out[b, dm] = predict(m, Xc)
        return out

    BLEND_REF = 0.5

    def blend(model_score, cells, w):
        """Rank average with the Au pathfinder index inside the belt domain (w = index weight);
        where there is no gold geochemistry the model's own rank is used."""
        if w <= 0:
            return model_score.astype("float64")
        out = np.zeros(len(cells), dtype="float64")
        d = dom[cells]
        if d.any():
            sm = rankdata(model_score[d]) / d.sum()
            ai = au_idx[cells][d]
            ok = np.isfinite(ai)
            sa = sm.copy()
            if ok.any():
                sa[ok] = rankdata(ai[ok]) / ok.sum()
            out[d] = (1 - w) * sm + w * sa
        return out

    def within_terrain(scores, ref, reg_cells, p_local, w_te):
        """Capture in the top 10% when each held-out label is ranked only against belt ground of its
        own terrain bin (random = 10%)."""
        tb, dm = tbin[reg_cells], dom[reg_cells]
        pb = tb[p_local]
        hit_m = hit_r = wsum = 0.0
        npos = 0
        for b in np.unique(pb[pb >= 0]):
            m = (pb == b) & dm[p_local]
            sel = np.nonzero((tb == b) & dm)[0]
            if not m.any() or len(sel) < 20:
                continue
            li = np.searchsorted(sel, p_local[m])
            w = w_te[m]
            cut = len(sel) * 0.9
            hit_m += w[tie_broken_rank(scores[sel], rng)[li] >= cut].sum()
            hit_r += w[tie_broken_rank(ref[sel], rng)[li] >= cut].sum()
            wsum += w.sum()
            npos += int(m.sum())
        return (hit_m / wsum, hit_r / wsum, npos) if wsum > 0 else (np.nan, np.nan, 0)

    # ---- leave-one-region-out
    cv_rows, imp_rows, bag_rows = [], [], []
    for k in range(args.regions):
        tr_p = pos_region != k
        models = fit_bags(tr_p, [bgc[bgr != k] for bgc, bgr in zip(bg_cells, bg_regions)], neg_region != k)

        reg_cells = np.nonzero(region == k)[0]
        s_bags = predict_dom(models, reg_cells)
        s_reg = s_bags.mean(0)
        b_reg = base_score[reg_cells]
        loc = {c: i for i, c in enumerate(reg_cells)}
        te_p = np.nonzero((pos_region == k) & ~is_gsi)[0]
        p_local = np.array([loc[c] for c in pos_cell[te_p]], dtype="int64")
        w_te = pos_w[te_p]

        row = {"region": k, "positives": len(te_p), "occurrences": int(pos["occ"].to_numpy()[te_p].sum()),
               "area_km2": round(len(reg_cells) * (cell / 1000) ** 2)}
        r_model, r_base = tie_broken_rank(combine(s_reg, reg_cells), rng), tie_broken_rank(b_reg, rng)
        r_ref2 = tie_broken_rank(ref2[reg_cells], rng)
        for p in CAPTURE_AT:
            row[f"model_top{p}"] = capture(r_model, p_local, w_te, p)
            row[f"base_top{p}"] = capture(r_base, p_local, w_te, p)
            row[f"pathf_top{p}"] = capture(r_ref2, p_local, w_te, p)
        g_local = np.nonzero(in_green[reg_cells])[0]
        gp = np.isin(p_local, g_local)
        if g_local.size and gp.any():
            gl = {c: i for i, c in enumerate(g_local)}
            gpl = np.array([gl[c] for c in p_local[gp]])
            row["within_green_top10"] = capture(tie_broken_rank(s_reg[g_local], rng), gpl, w_te[gp], 10)
            ai = au_idx[reg_cells][g_local]
            row["within_green_pathf_top10"] = capture(tie_broken_rank(np.where(np.isfinite(ai), ai, -50.0), rng),
                                                      gpl, w_te[gp], 10)
            row["green_positives"] = int(gp.sum())
        else:
            row["within_green_top10"], row["within_green_pathf_top10"], row["green_positives"] = np.nan, np.nan, 0
        if tbin is not None:
            ai_reg = au_idx[reg_cells]
            tm, tr_, tw = within_terrain(s_reg, np.where(np.isfinite(ai_reg), ai_reg, -50.0), reg_cells, p_local, w_te)
            row["terrain_top10"], row["terrain_pathf_top10"], row["terrain_positives"] = tm, tr_, tw
        s_bl = blend(s_reg, reg_cells, BLEND_REF)
        r_bl = tie_broken_rank(combine(s_bl, reg_cells), rng)
        row["blend_top1"], row["blend_top10"] = capture(r_bl, p_local, w_te, 1), capture(r_bl, p_local, w_te, 10)
        row["blend_green10"] = (capture(tie_broken_rank(s_bl[g_local], rng), gpl, w_te[gp], 10)
                                if g_local.size and gp.any() else np.nan)
        if tbin is not None:
            row["blend_terrain10"] = within_terrain(s_bl, np.where(np.isfinite(ai_reg), ai_reg, -50.0),
                                                    reg_cells, p_local, w_te)[0]
        if n_bags > 1:   # how much would a single bag have varied?
            for b in range(n_bags):
                br = {"region": k, "bag": b, "positives": len(te_p), "green_positives": row["green_positives"],
                      "model_top1": capture(tie_broken_rank(combine(s_bags[b], reg_cells), rng), p_local, w_te, 1)}
                br["within_green_top10"] = (capture(tie_broken_rank(s_bags[b][g_local], rng), gpl, w_te[gp], 10)
                                            if g_local.size and gp.any() else np.nan)
                bag_rows.append(br)

        te_b = np.nonzero(bg_regions[0] == k)[0]
        Xte = np.vstack([X[pos_cell[te_p]], X[bg_cells[0][te_b]]])
        yte = np.r_[np.ones(len(te_p)), np.zeros(len(te_b))]
        wte = np.r_[w_te, np.ones(len(te_b))]

        def ens(Xa):
            return np.mean([predict(m, Xa) for m in models], axis=0)
        if args.learner == "cnn":
            te_cells = np.r_[pos_cell[te_p], bg_cells[0][te_b]]
            auc = weighted_auc(yte, np.mean([m[rows[te_cells], cols[te_cells]] for m in models], axis=0), wte)
        else:
            auc = weighted_auc(yte, ens(Xte), wte)
        row["auc"] = auc
        cv_rows.append(row)

        # group importance: shuffle each group's columns jointly, measure AUC drop of the averaged model
        for gname, prefix in (GROUPS if args.learner == "hgb" else []):
            cols_g = [j for j, nm in enumerate(names) if nm.startswith(prefix)]
            if not cols_g or not np.isfinite(auc):
                continue
            drops = []
            for _ in range(5):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xp))
                Xp[:, cols_g] = Xte[perm][:, cols_g]
                drops.append(auc - weighted_auc(yte, ens(Xp), wte))
            imp_rows.append({"region": k, "group": gname, "auc_drop": float(np.mean(drops))})
        wg = row["within_green_top10"]
        wg_txt = "n/a" if np.isnan(wg) else f"{wg:.0%}"
        print(f"  region {k}: {row['positives']:3} positives  AUC {auc:.2f}  "
              f"top10 model {row['model_top10']:.0%} vs greenstone {row['base_top10']:.0%}  "
              f"within-greenstone top10 {wg_txt}  [{time.time() - t0:.0f}s]", flush=True)

    # ---- final models on everything, full map
    models = fit_bags(np.ones(len(pos_cell), dtype=bool), bg_cells)
    all_cells = np.arange(n)
    s_bags = predict_dom(models, all_cells)

    def to_pct(model_score):
        score = combine(model_score, all_cells)
        pct = ((rankdata(score, method="average") - 1) / max(n - 1, 1) * 100).astype("float32")  # ties share a rank
        g = np.full(shape, np.nan, dtype="float32")
        g[rows, cols] = pct
        return g

    prof = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="float32", crs=spec["crs"],
                transform=T, nodata=np.nan, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    grid_pct = to_pct(blend(s_bags.mean(0), all_cells, args.blend_au))
    with rasterio.open(out / "prospectivity_pct.tif", "w", **prof) as dst:
        dst.write(grid_pct, 1)
    if n_bags > 1:
        (out / "bags").mkdir(exist_ok=True)
        for old in (out / "bags").glob("pct_bag*.tif"):
            old.unlink()
        for b in range(n_bags):
            with rasterio.open(out / "bags" / f"pct_bag{b + 1:02d}.tif", "w", **prof) as dst:
                dst.write(to_pct(blend(s_bags[b], all_cells, args.blend_au)), 1)

    # ---- reports
    keys = list(cv_rows[0].keys())
    with open(out / "cv_regions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in cv_rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    def wmean(key, weight="positives"):
        vals = np.array([r[key] for r in cv_rows], dtype="float64")
        ws = np.array([r[weight] for r in cv_rows], dtype="float64")
        ok = np.isfinite(vals) & (ws > 0)
        return float((vals[ok] * ws[ok]).sum() / ws[ok].sum()) if ok.any() else np.nan

    summary = {f"model_top{p}": wmean(f"model_top{p}") for p in CAPTURE_AT}
    summary.update({f"base_top{p}": wmean(f"base_top{p}") for p in CAPTURE_AT})
    summary.update({f"pathf_top{p}": wmean(f"pathf_top{p}") for p in CAPTURE_AT})
    summary["within_green_top10"] = wmean("within_green_top10", "green_positives")
    summary["within_green_pathf_top10"] = wmean("within_green_pathf_top10", "green_positives")
    summary["auc"] = wmean("auc")
    summary["blend_top1"], summary["blend_top10"] = wmean("blend_top1"), wmean("blend_top10")
    summary["blend_green10"] = wmean("blend_green10", "green_positives")
    if tbin is not None:
        summary["blend_terrain10"] = wmean("blend_terrain10", "terrain_positives")
    if tbin is not None:
        summary["terrain_top10"] = wmean("terrain_top10", "terrain_positives")
        summary["terrain_pathf_top10"] = wmean("terrain_pathf_top10", "terrain_positives")
        top1 = np.nonzero(dom & (grid_pct[rows, cols] >= 99.0))[0]
        summary["terrain_mix"] = {
            name: {"median_relief_m": float(np.nanmedian(relief_c[c])), "share_mostly_scrub": float(np.nanmean(scrub_c[c] >= 0.5))}
            for name, c in (("labels", pos_cell), ("belt", np.nonzero(dom)[0]), ("top1pct", top1))}
    bag_spread = {}
    if bag_rows:
        for key, wkey in (("model_top1", "positives"), ("within_green_top10", "green_positives")):
            per_bag = []
            for b in range(n_bags):
                rs = [r for r in bag_rows if r["bag"] == b]
                v = np.array([r[key] for r in rs], dtype="float64")
                ws = np.array([r[wkey] for r in rs], dtype="float64")
                ok = np.isfinite(v) & (ws > 0)
                per_bag.append(float((v[ok] * ws[ok]).sum() / ws[ok].sum()) if ok.any() else np.nan)
            bag_spread[key] = {"min": float(np.nanmin(per_bag)), "max": float(np.nanmax(per_bag)),
                               "per_bag": [round(x, 3) for x in per_bag]}

    imp = {}
    for r in imp_rows:
        imp.setdefault(r["group"], []).append(r["auc_drop"])
    imp_mean = sorted(((g, float(np.mean(v))) for g, v in imp.items()), key=lambda t: -t[1])
    with open(out / "group_importance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "mean_auc_drop"])
        for g, v in imp_mean:
            w.writerow([g, f"{v:.4f}"])

    run = {"name": args.name, "learner": args.learner, "domain": args.domain, "belt_km": args.belt_km, "excluded_prefixes": excl, "only": only,
           "monotonic": bool(args.monotonic), "features": names, "regions": args.regions,
           "region_seed": args.region_seed, "seed": args.seed, "bags": n_bags, "bag_spread": bag_spread, **gsi_info,
           "match_terrain": args.match_terrain, "blend_au": args.blend_au,
           "terrain_correct": corr_pref, "terrain_r2": terrain_r2,
           "background": int(len(bg_cells[0])), "positives": int(len(pos_cell)),
           "labels_outside_mask": missing, "model_params": MODEL_PARAMS, "summary": summary,
           "group_importance": dict(imp_mean)}
    (out / "run.json").write_text(json.dumps(run, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 9), dpi=120)
        im = ax.imshow(grid_pct[::2, ::2], cmap="magma", vmin=50, vmax=100, interpolation="nearest")
        oc = lab[(lab.cell >= 0) & (lab.label_class == "occurrence")]
        ow = lab[(lab.cell >= 0) & (lab.label_class != "occurrence")]
        ax.scatter(lc_[ow.index] / 2, lr[ow.index] / 2, s=10, marker="^", facecolor="none", edgecolor="cyan", lw=0.6, label="old workings")
        ax.scatter(lc_[oc.index] / 2, lr[oc.index] / 2, s=14, facecolor="none", edgecolor="lime", lw=0.8, label="gold occurrence")
        ax.set_axis_off()
        ax.legend(loc="lower left", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, label="prospectivity percentile (50-100 shown)")
        ax.set_title(f"Gold prospectivity V1 ({args.name})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "prospectivity.png")
    except ImportError:
        pass

    print(f"\n=== {args.name}: leave-one-region-out ({args.regions} regions), weighted by positives ===")
    print(f"{'':22}{'top1%':>8}{'top5%':>8}{'top10%':>8}{'top20%':>8}")
    print(f"{'model':22}" + "".join(f"{summary[f'model_top{p}']:>8.0%}" for p in CAPTURE_AT))
    print(f"{'greenstone baseline':22}" + "".join(f"{summary[f'base_top{p}']:>8.0%}" for p in CAPTURE_AT))
    print(f"{'greenstone + Au index':22}" + "".join(f"{summary[f'pathf_top{p}']:>8.0%}" for p in CAPTURE_AT))
    print(f"{'random':22}" + "".join(f"{p / 100:>8.0%}" for p in CAPTURE_AT))
    print("\nWithin greenstone, top 10% of belt area captures of held-out gold (random = 10%):")
    print(f"  model {summary['within_green_top10']:.0%}   |   Au pathfinder index alone {summary['within_green_pathf_top10']:.0%}")
    if tbin is not None:
        tmx = summary["terrain_mix"]
        print("Within the same terrain (relief quintile x scrub third), top 10% captures (random = 10%):")
        print(f"  model {summary['terrain_top10']:.0%}   |   Au pathfinder index alone {summary['terrain_pathf_top10']:.0%}")
        print("  terrain of the model's top 1%: median relief {:.0f} m, mostly scrub {:.0%}  "
              "(labels {:.0f} m / {:.0%}, belt ground {:.0f} m / {:.0%})".format(
                  tmx["top1pct"]["median_relief_m"], tmx["top1pct"]["share_mostly_scrub"],
                  tmx["labels"]["median_relief_m"], tmx["labels"]["share_mostly_scrub"],
                  tmx["belt"]["median_relief_m"], tmx["belt"]["share_mostly_scrub"]))
    print("Model + Au pathfinder index, 50/50 rank blend (reference): top1% {:.0%}, within greenstone {:.0%}{}".format(
        summary["blend_top1"], summary["blend_green10"],
        f", within terrain {summary['blend_terrain10']:.0%}" if tbin is not None else ""))
    if args.blend_au > 0:
        print(f"  final map = blend with index weight {args.blend_au:g}"
              + (" (its held-out scores are the blend line above)" if abs(args.blend_au - BLEND_REF) < 1e-9 else
                 " (held-out scores above are for 0.5; this weight was not cross-validated)"))
    if bag_spread:
        bs = bag_spread
        print(f"  single bags would have ranged: top1% {bs['model_top1']['min']:.0%}-{bs['model_top1']['max']:.0%}, "
              f"within-greenstone top10 {bs['within_green_top10']['min']:.0%}-{bs['within_green_top10']['max']:.0%} "
              f"(average of {n_bags} reported above)")
    print(f"Held-out AUC (positives vs background): {summary['auc']:.2f}")
    print("\nFeature-group importance (held-out AUC drop when shuffled):")
    for g, v in imp_mean:
        print(f"  {g:14} {v:+.3f}")
    print(f"\nOutputs in {out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
