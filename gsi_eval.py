#!/usr/bin/env python3
"""
gsi_eval.py - blind test of the prospectivity maps against GSI's own sampling results.

Run from the repository folder after gsi_labels.py:
    python gsi_eval.py data\\processed
    python gsi_eval.py data\\processed --models pg_ens pg_rc --blend pg_ens,pg_rc

None of the GSI sites were used in training, so this is an independent check: where GSI found gold
(trench, channel, drill, rock >= 0.5 g/t; anomalous 0.1-0.5 g/t) and where it tested ground and found
nothing (barren: >= 3 rock, trench or drill samples all < 0.1 g/t), how does each map rank those sites?

Reads   data\\processed\\gsi_label_sites.csv               (from gsi_labels.py)
        data\\processed\\models\\<model>\\prospectivity_pct.tif (and bags\\pct_bag*.tif if present)
        data\\processed\\mask_model.tif, stack\\geo_greenstone_sdist.tif, stack\\gc_au_index.tif
Writes  data\\processed\\models\\gsi_eval.csv     one row per map: shares in the top 10%, AUC, medians
        data\\processed\\gsi_eval_sites.csv        every GSI site with its percentile on each map
        data\\processed\\gsi_eval.png              the sites on each map's belt-ground percentile scale

How it is scored: each site is ranked against all belt ground (within 1 km of greenstone, --belt-km to
widen), the same ground the models rank, so 90 = in the top 10% of belt ground. Sites off belt ground are counted
separately (every map ranks them below the belts). References that need no training: the Au
pathfinder index alone, and a 50/50 rank blend of a model with the index (as train_v1 scores it).
  AUC gold vs barren   chance that a random GSI gold site ranks above a random GSI barren site
                       (0.5 = no better than a coin; this is the number to compare between maps)
  independent          the same, using only sites more than --indep-km from every training label.
                       Gold sites next to a training label are not a fair test of a trained model
                       (it learned those spots), so judge models on this column.
  barren in top 10%    share of GSI-tested barren sites that the map puts in its top 10%. GSI chose
                       these spots because they looked prospective, so this is expected to be above
                       10% for any sensible map; lower than the other maps is better.
Small numbers: the sites come from a dozen reports and cluster in a few blocks, so uncertainty is
measured by resampling whole clusters (sites within --cluster-km): '90% range' is the spread of the
independent AUC, 'beats index' the share of resamples in which a map scores above the Au index alone.
The bag range shows how much a model's score moves with its background draw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
    from scipy.stats import rankdata
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

BELT_BUFFER_M = 1000.0
GOLD = ("positive", "anomalous")
BARREN = ("barren_strong", "barren")


def read1(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def belt_percentile(grid: np.ndarray, belt: np.ndarray, rr: np.ndarray, cc: np.ndarray, ok: np.ndarray,
                    window: int) -> np.ndarray:
    """Percentile (0-100, ties share the mid rank) of each site's value among all belt cells."""
    v = grid[belt].astype("float64")
    v = np.where(np.isfinite(v), v, -np.inf)
    s = np.sort(v)
    out = np.full(len(rr), np.nan)
    for i in np.nonzero(ok)[0]:
        r0, c0 = rr[i], cc[i]
        if window:
            sl = (slice(max(r0 - window, 0), r0 + window + 1), slice(max(c0 - window, 0), c0 + window + 1))
            w = grid[sl][belt[sl]]
            w = w[np.isfinite(w)]
            x = float(w.max()) if len(w) else -np.inf
        else:
            x = float(grid[r0, c0]) if np.isfinite(grid[r0, c0]) else -np.inf
        lo, hi = np.searchsorted(s, x, "left"), np.searchsorted(s, x, "right")
        out[i] = (lo + hi) / 2 / len(s) * 100
    return out


def blend_grid(model: np.ndarray, au: np.ndarray, belt: np.ndarray, w: float = 0.5) -> np.ndarray:
    """train_v1's blend: rank average inside belt ground; where the index is missing the model's rank is used."""
    out = np.full(model.shape, np.nan, dtype="float64")
    m = model[belt].astype("float64")
    sm = rankdata(np.where(np.isfinite(m), m, -np.inf)) / len(m)
    a = au[belt].astype("float64")
    ok = np.isfinite(a)
    sa = sm.copy()
    if ok.any():
        sa[ok] = rankdata(a[ok]) / ok.sum()
    out[belt] = (1 - w) * sm + w * sa
    return out


def site_clusters(x: np.ndarray, y: np.ndarray, d_m: float) -> np.ndarray:
    """Single-linkage clusters of sites closer than d_m (metres): the units the bootstrap resamples."""
    n = len(x)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        d = np.hypot(x[i + 1:] - x[i], y[i + 1:] - y[i])
        for j in np.nonzero(d <= d_m)[0] + i + 1:
            parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def bootstrap(site_pct: dict, gold: np.ndarray, barren: np.ndarray, cl: np.ndarray, n_boot: int, ref: str,
              seed: int = 0) -> dict:
    """Resample whole clusters with replacement; AUC (gold vs barren) per map per resample.
    Returns per map: 5th / 95th percentile of the AUC and the share of resamples in which it beats `ref`."""
    use = gold | barren
    groups = [np.nonzero(use & (cl == c))[0] for c in np.unique(cl[use])]
    rng = np.random.default_rng(seed)
    aucs = {k: [] for k in site_pct}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        g, b = idx[gold[idx]], idx[barren[idx]]
        for k, v in site_pct.items():
            aucs[k].append(auc(v[g], v[b]) if len(g) and len(b) else np.nan)
    out, a_ref = {}, np.array(aucs[ref])
    for k, a in aucs.items():
        a = np.array(a)
        ok = np.isfinite(a) & np.isfinite(a_ref)
        out[k] = {"auc_indep_p05": float(np.nanpercentile(a, 5)) if np.isfinite(a).any() else np.nan,
                  "auc_indep_p95": float(np.nanpercentile(a, 95)) if np.isfinite(a).any() else np.nan,
                  "share_beats_au_index": (float(np.mean((a[ok] > a_ref[ok]) + 0.5 * (a[ok] == a_ref[ok])))
                                           if ok.any() and k != ref else np.nan)}
    return out


def score(pct: np.ndarray, cls: np.ndarray, indep: np.ndarray) -> dict:
    """Metrics for one map over sites on belt ground (pct not NaN)."""
    on = np.isfinite(pct)
    res = {}
    for name, labs in (("positive", ("positive",)), ("anomalous", ("anomalous",)), ("barren", BARREN),
                       ("barren_weak", ("barren_weak",))):
        sel = on & np.isin(cls, labs)
        res[f"n_{name}"] = int(sel.sum())
        res[f"{name}_top10"] = float((pct[sel] >= 90).mean()) if sel.any() else float("nan")
        res[f"{name}_median_pct"] = float(np.median(pct[sel])) if sel.any() else float("nan")
    g, b = on & np.isin(cls, GOLD), on & np.isin(cls, BARREN)
    res["auc_gold_vs_barren"] = auc(pct[g], pct[b])
    res["auc_independent"] = auc(pct[g & indep], pct[b & indep])
    res["n_gold_indep"], res["n_barren_indep"] = int((g & indep).sum()), int((b & indep).sum())
    res["auc_gold_vs_weak"] = auc(pct[g], pct[on & (cls == "barren_weak")])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Blind test of prospectivity maps against GSI gold / barren sites.")
    ap.add_argument("processed")
    ap.add_argument("--models", nargs="*", default=None,
                    help="model folders under models\\ to test (default: every folder with prospectivity_pct.tif)")
    ap.add_argument("--blend", default="pg_ens,pg_rc", help="models to also test as a 50/50 blend with the Au index")
    ap.add_argument("--indep-km", type=float, default=1.0,
                    help="'independent' sites are farther than this from every training label (default 1 km)")
    ap.add_argument("--belt-km", type=float, default=BELT_BUFFER_M / 1000,
                    help="rank sites against ground within this distance of greenstone (default 1 km, as the models)")
    ap.add_argument("--tag", default=None, help="suffix for the output files (default: _belt<N>km when --belt-km is not 1)")
    ap.add_argument("--sites-file", default="",
                    help="score only these sites (e.g. models\\<model>\\gsi_holdout_sites.csv from train_v1 --gsi-holdout)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples of site clusters (default 2000)")
    ap.add_argument("--cluster-km", type=float, default=5.0,
                    help="sites closer than this form one cluster for the bootstrap (default 5 km)")
    ap.add_argument("--window", type=int, default=0,
                    help="use the best cell within this many cells of the site (0 = the site's own cell)")
    args = ap.parse_args()
    proc = Path(args.processed)
    mroot = proc / "models"
    sites_p = proc / "gsi_label_sites.csv"
    if not sites_p.exists():
        sys.exit(f"{sites_p} not found: run gsi_labels.py first")

    with rasterio.open(proc / "mask_model.tif") as src:
        mask = src.read(1) == 1
        T, crs = src.transform, src.crs
    with rasterio.open(proc / "stack" / "geo_greenstone_sdist.tif") as src:
        sd = src.read(1).astype("float64")
    belt = mask & np.isfinite(sd) & (sd <= args.belt_km * 1000)
    tag = args.tag if args.tag is not None else ("" if args.belt_km == 1 else f"_belt{args.belt_km:g}km")
    with rasterio.open(proc / "stack" / "gc_au_index.tif") as src:
        au = src.read(1).astype("float64")
    au = np.where(mask, au, np.nan)
    print(f"Belt ground (within {args.belt_km:g} km of greenstone): {int(belt.sum()):,} cells")

    sites = pd.read_csv(sites_p)
    sites = sites[sites["label"] != "too_few_samples"].reset_index(drop=True)
    if args.sites_file:   # held-out fold only: match on position
        keep_ = pd.read_csv(args.sites_file)
        kk = set(zip(keep_["lon"].round(5), keep_["lat"].round(5)))
        sites = sites[[k in kk for k in zip(sites["lon"].round(5), sites["lat"].round(5))]].reset_index(drop=True)
        print(f"Scoring only the {len(sites)} sites listed in {args.sites_file}")
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xy = np.array([tr.transform(lo, la) for lo, la in zip(sites["lon"], sites["lat"])])
    rr, cc = rowcol(T, xy[:, 0], xy[:, 1])
    rr, cc = np.asarray(rr), np.asarray(cc)
    ing = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    on_belt = np.zeros(len(sites), dtype=bool)
    on_belt[ing] = belt[rr[ing], cc[ing]]
    d_green = np.full(len(rr), np.nan)
    d_green[ing] = sd[rr[ing], cc[ing]] / 1000.0                  # km outside greenstone (negative = inside)
    cls = sites["label"].to_numpy()
    dist = pd.to_numeric(sites.get("dist_known_gold_km"), errors="coerce").to_numpy()
    indep = np.nan_to_num(dist, nan=0.0) > args.indep_km

    print("\nGSI sites (too_few_samples left out):")
    for lab in ("positive", "anomalous", "barren_strong", "barren", "barren_weak"):
        sel = cls == lab
        print(f"  {lab:14} {int(sel.sum()):4}   on belt ground {int((sel & on_belt).sum()):4}   "
              f"of those > {args.indep_km:g} km from training labels {int((sel & on_belt & indep).sum()):4}")
    nb_rep = sites.loc[np.isin(cls, BARREN) & on_belt, "reports"].astype(str).str.split(",").explode().nunique()
    print(f"  barren sites on belt ground come from {nb_rep} reports")
    off = np.isin(cls, GOLD) & ~on_belt
    if off.any():
        print("  gold sites off belt ground (every map ranks these below all belt ground):")
        o = sites[off].assign(d=d_green[off]).sort_values("best_au_ppb", ascending=False)
        for _, r in o.head(12).iterrows():
            print(f"    {r['best_au_ppb'] / 1000:6.2f} g/t  {r['lon']:.4f} {r['lat']:.4f}  {r['d']:.1f} km from greenstone  ({r['reports']})")

    # ---- the maps
    models = args.models
    if not models:
        models = sorted(d.name for d in mroot.iterdir() if (d / "prospectivity_pct.tif").exists()) if mroot.exists() else []
    models = [m for m in models if (mroot / m / "prospectivity_pct.tif").exists()]
    blends = [b.strip() for b in args.blend.split(",") if b.strip() in models]
    maps = [("Au index alone", "au_index", lambda: au, [])]
    for m in models:
        bags = sorted((mroot / m / "bags").glob("pct_bag*.tif"))
        maps.append((m, m, (lambda m=m: read1(mroot / m / "prospectivity_pct.tif")), bags))
        if m in blends:
            maps.append((f"{m} + Au index", f"{m}+au",
                         (lambda m=m: blend_grid(read1(mroot / m / "prospectivity_pct.tif"), au, belt)),
                         [("blend", b) for b in bags]))
    rows, site_pct = [], {}
    for label, key, load, bags in maps:
        g = load().astype("float64")
        g = np.where(belt, g, np.nan)
        pct = np.full(len(sites), np.nan)
        pct[on_belt] = belt_percentile(g, belt, rr, cc, on_belt, args.window)[on_belt]
        site_pct[key] = pct
        r = {"map": label, **score(pct, cls, indep)}
        if bags:   # spread over the background draws
            vals = {"auc_independent": [], "auc_gold_vs_barren": [], "barren_top10": []}
            for b in bags:
                gb = blend_grid(read1(b[1]), au, belt) if isinstance(b, tuple) else read1(b).astype("float64")
                gb = np.where(belt, gb, np.nan)
                pb = np.full(len(sites), np.nan)
                pb[on_belt] = belt_percentile(gb, belt, rr, cc, on_belt, args.window)[on_belt]
                sb = score(pb, cls, indep)
                for k in vals:
                    vals[k].append(sb[k])
            for k, v in vals.items():
                v = [x for x in v if x == x]
                r[f"{k}_bag_min"], r[f"{k}_bag_max"] = (min(v), max(v)) if v else (np.nan, np.nan)
        rows.append(r)
    res = pd.DataFrame(rows)
    # ---- uncertainty: the sites come in a few clusters, so resample clusters, not sites
    cl = site_clusters(xy[:, 0], xy[:, 1], args.cluster_km * 1000)
    g_ind = on_belt & indep & np.isin(cls, GOLD)
    b_ind = on_belt & indep & np.isin(cls, BARREN)
    n_cg, n_cb = len(np.unique(cl[g_ind])), len(np.unique(cl[b_ind]))
    bs = bootstrap(site_pct, g_ind, b_ind, cl, args.boot, "au_index")
    for col in ("auc_indep_p05", "auc_indep_p95", "share_beats_au_index"):
        res[col] = [bs[k][col] for k in site_pct]
    res.to_csv(mroot / f"gsi_eval{tag}.csv", index=False)
    out_sites = sites[["label", "lon", "lat", "best_au_ppb", "best_sample", "reports", "dist_known_gold_km"]].copy()
    out_sites["on_belt"] = on_belt
    out_sites["independent"] = indep
    out_sites["cluster"] = cl
    out_sites["dist_greenstone_km"] = np.round(d_green, 2)
    for k, v in site_pct.items():
        out_sites[f"pct_{k}"] = np.round(v, 1)
    out_sites.to_csv(proc / f"gsi_eval_sites{tag}.csv", index=False, encoding="utf-8-sig")

    # ---- console table
    def f(x, pct=False):
        return "  n/a" if x != x else (f"{x:5.0%}" if pct else f"{x:5.2f}")
    print("\nOn belt ground; 'top 10%' = share of those sites in the map's top 10% of belt ground (random 10%).")
    print(f"{'':24}{'-- share in the top 10% --':>27}   {'---------- gold vs barren AUC ----------':>45}")
    print(f"{'map':24}{'>=0.5':>6}{'0.1-0.5':>8}{'barren':>7}{'weak':>6}   {'all':>5}{'indep':>7}{'90% range':>11}"
          f"{'beats index':>13}{'bags':>11}")
    for _, r in res.iterrows():
        bag = (f"{r['auc_independent_bag_min']:.2f}-{r['auc_independent_bag_max']:.2f}"
               if "auc_independent_bag_min" in r and r["auc_independent_bag_min"] == r["auc_independent_bag_min"] else "")
        rng90 = f"{r['auc_indep_p05']:.2f}-{r['auc_indep_p95']:.2f}" if r["auc_indep_p05"] == r["auc_indep_p05"] else ""
        beat = f"{r['share_beats_au_index']:.0%}" if r["share_beats_au_index"] == r["share_beats_au_index"] else "-"
        print(f"{r['map'][:24]:24}{f(r['positive_top10'], 1):>6}{f(r['anomalous_top10'], 1):>8}{f(r['barren_top10'], 1):>7}"
              f"{f(r['barren_weak_top10'], 1):>6}   {f(r['auc_gold_vs_barren']):>5}{f(r['auc_independent']):>7}{rng90:>11}"
              f"{beat:>13}{bag:>11}")
    r0 = res.iloc[0]
    print(f"\nSites: gold {int(r0['n_positive'] + r0['n_anomalous'])} (>= 0.5 g/t {int(r0['n_positive'])}), "
          f"barren {int(r0['n_barren'])}; independent: gold {int(r0['n_gold_indep'])}, barren {int(r0['n_barren_indep'])}")
    if r0["n_gold_indep"] < 5:
        print("  Fewer than 5 independent gold sites: the 'AUC indep' column is too thin to choose a model on; more GIS files will fix that.")
    print("gold = >= 0.1 g/t sites (both columns); weak = soil/stream-only sites < 50 ppb (not in the AUC).")
    print(f"Independent gold sites fall in {n_cg} clusters and barren sites in {n_cb} (sites within {args.cluster_km:g} km "
          "grouped); the 90% range resamples whole clusters.")
    print("'beats index' = share of those resamples in which the map's independent AUC is above the Au index alone's;"
          " above ~90% is a real difference, 30-70% is a tie.")
    print("AUC 0.5 = no better than a coin at telling GSI gold sites from GSI barren sites.")

    # ---- figure: each map's percentile scale with the sites on it
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ink, ink2, grid_c, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
        style = [("barren", BARREN, "#2a78d6", "s", 22), ("anomalous (0.1-0.5 g/t)", ("anomalous",), "#1baf7a", "^", 34),
                 ("gold >= 0.5 g/t", ("positive",), "#eb6834", "*", 110)]
        keys = list(site_pct)
        fig, ax = plt.subplots(figsize=(9, 0.75 * len(keys) + 1.6), dpi=130, facecolor=surf)
        ax.set_facecolor(surf)
        rng = np.random.default_rng(0)
        for yi, k in enumerate(keys):
            for name, labs, col, mk, sz in style:
                sel = np.isin(cls, labs) & np.isfinite(site_pct[k])
                y = len(keys) - 1 - yi + (rng.uniform(-0.22, 0.22, sel.sum()) if name == "barren" else
                                          rng.uniform(-0.12, 0.12, sel.sum()))
                ax.scatter(site_pct[k][sel], y, s=sz, c=col, marker=mk, edgecolors=surf, linewidths=0.8,
                           label=name if yi == 0 else None, zorder=3 if name != "barren" else 2)
            a = res.iloc[yi]["auc_independent"]
            ax.text(101.5, len(keys) - 1 - yi, f"AUC {res.iloc[yi]['auc_gold_vs_barren']:.2f}"
                    + (f" | indep {a:.2f}" if a == a else ""), va="center", fontsize=8, color=ink2)
        ax.axvline(90, color=ink2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(90.5, len(keys) - 0.45, "top 10%", fontsize=8, color=ink2)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([res.iloc[len(keys) - 1 - i]["map"] for i in range(len(keys))], fontsize=8.5, color=ink)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(keys) - 0.3)
        ax.set_xlabel("percentile among belt ground (100 = most prospective)", fontsize=8.5, color=ink2)
        ax.grid(axis="x", color=grid_c, lw=0.8)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(grid_c)
        ax.tick_params(colors=ink2, labelsize=8, length=0)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, fontsize=8, frameon=False)
        ax.set_title("GSI sampling results ranked on each map (none were used in training)", fontsize=10, color=ink, loc="left")
        fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2 if len(keys) > 3 else 0.3)
        fig.savefig(proc / f"gsi_eval{tag}.png", facecolor=surf)
        plt.close(fig)
    except Exception as ex:
        print(f"  (figure skipped: {type(ex).__name__}: {ex})")
    print(f"\nWritten {mroot / f'gsi_eval{tag}.csv'}, {proc / f'gsi_eval_sites{tag}.csv'}, {proc / f'gsi_eval{tag}.png'}")


if __name__ == "__main__":
    main()
