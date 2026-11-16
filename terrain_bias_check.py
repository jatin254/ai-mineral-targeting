#!/usr/bin/env python3
"""
terrain_bias_check.py - is the model finding gold, or finding hills?

Why: the desk check put most targets on scrub-covered hills, and radiometrics (which read higher
over exposed rock than over soil) is one of the strongest feature groups. Known gold was also mostly
found by prospectors where rock is exposed. If the gold labels sit on hills more than typical belt
ground does, the model can learn "exposed rock" instead of "gold", and will under-rank soil-covered
ground, which is where undiscovered deposits are most likely to be. This script measures that.

Run from the repository folder (needs internet, about 3-8 minutes, cached after that):
    python terrain_bias_check.py data/processed --model pg_ens

Compares three sets of 300 m circles on the same open data the desk check uses (Copernicus DEM,
ESA WorldCover):
    gold labels      every gold occurrence and old working inside the belt domain
    belt background  random belt ground at least 500 m from any label (what the model trains against)
    top 1% ground    random cells the model ranks in its top 1%
Writes models/<model>/terrain_check/
    terrain_sites.csv     every circle: group, relief, slope, scrub/bare and cropland share, model
                          percentile, radiometric values
    terrain_summary.txt   the comparison and a plain verdict (also printed)
    terrain_check.png     relief by group, and model percentile against relief on belt ground
How to read it
    labels hillier than background   detection bias in the labels. The V2 fix is a background sampled
                                     to match the labels' terrain ("target-group background", Phillips
                                     et al. 2009, Ecological Applications 19:181-197), so terrain alone
                                     cannot separate gold from background.
    top 1% hillier than the labels   the model favours hills beyond what the gold justifies: likely a
                                     radiometric or terrain artefact.
    neither                          no terrain bias found; the hill-heavy target list reflects geology
                                     (resistant greenstone ridges).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_desk_check as adc  # noqa: E402  (sets the GDAL network options before rasterio loads)

try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.transform import Affine, rowcol
    from scipy import ndimage
    from scipy.stats import mannwhitneyu, spearmanr
    from shapely.geometry import Point
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

BELT_BUFFER_M = 1000.0
LABEL_CLEAR_M = 500.0
HILLY_M = 25.0          # same relief cut-off the desk check uses for hill ground
RAD = ("rad_k", "rad_eth", "rad_eu", "rad_dose")


def read(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        return src.read(1).astype("float64")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare terrain at gold labels, belt background and top targets.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("--model", default="pg_ens")
    ap.add_argument("--background", type=int, default=300, help="random belt circles")
    ap.add_argument("--top", type=int, default=150, help="random top-1%% circles")
    ap.add_argument("--radius", type=float, default=300.0, help="circle radius, m")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    proc = Path(args.processed)
    mdir = proc / "models" / args.model
    out = mdir / "terrain_check"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    spec = json.loads((proc / "grid_spec.json").read_text())
    cell = float(spec["cell_m"])
    T = Affine(cell, 0, spec["xmin"], 0, -cell, spec["ymax"])
    mask = read(proc / "mask_model.tif") == 1
    sdist = read(proc / "stack" / "geo_greenstone_sdist.tif")
    belt = mask & np.isfinite(sdist) & (sdist <= BELT_BUFFER_M)
    pct = read(mdir / "prospectivity_pct.tif")
    rad = {k: read(proc / "stack" / f"{k}.tif") for k in RAD if (proc / "stack" / f"{k}.tif").exists()}

    # ---- sites
    lab = gpd.read_file(proc / args.gpkg, layer="gold_labels").to_crs(spec["crs"])
    lr, lc = rowcol(T, lab.geometry.x.values, lab.geometry.y.values)
    lr, lc = np.asarray(lr), np.asarray(lc)
    ok = (lr >= 0) & (lr < belt.shape[0]) & (lc >= 0) & (lc < belt.shape[1])
    ok[ok] = belt[lr[ok], lc[ok]]
    labels = pd.DataFrame({"group": "gold label", "site": lab.loc[ok, "label_id"].astype(str).values,
                           "row": lr[ok], "col": lc[ok]})
    pres = np.zeros(belt.shape, dtype=bool)
    pres[lr[ok], lc[ok]] = True
    far = ndimage.distance_transform_edt(~pres) * cell >= LABEL_CLEAR_M
    bg_r, bg_c = np.nonzero(belt & far)
    pick = rng.choice(len(bg_r), size=min(args.background, len(bg_r)), replace=False)
    bg = pd.DataFrame({"group": "belt background", "site": [f"B{i:03d}" for i in range(len(pick))],
                       "row": bg_r[pick], "col": bg_c[pick]})
    tp_r, tp_c = np.nonzero(belt & (pct >= 99.0))
    pick = rng.choice(len(tp_r), size=min(args.top, len(tp_r)), replace=False)
    top = pd.DataFrame({"group": "top 1% ground", "site": [f"T{i:03d}" for i in range(len(pick))],
                        "row": tp_r[pick], "col": tp_c[pick]})
    sites = pd.concat([labels, bg, top], ignore_index=True)
    sites["x"] = spec["xmin"] + (sites["col"] + 0.5) * cell
    sites["y"] = spec["ymax"] - (sites["row"] + 0.5) * cell
    sites["model_pct"] = pct[sites["row"], sites["col"]]
    for k, a in rad.items():
        sites[k] = a[sites["row"], sites["col"]]
    print(f"Sites: {len(labels)} gold labels in the belt domain, {len(bg)} belt background, {len(top)} top 1%")

    # ---- measure
    b = adc.ll(adc.box(sites["x"].min(), sites["y"].min(), sites["x"].max(), sites["y"].max())).bounds
    use = {"wc": adc.reachable(adc.WORLDCOVER_URL.format(tile=adc.worldcover_tiles(b)[0])),
           "dem": adc.reachable(adc.DEM_URL.format(tile=adc.dem_tiles(b)[0]))}
    print("Sources: " + ", ".join(f"{n} {'ok' if use[k] else 'unavailable'}"
                                  for k, n in (("wc", "WorldCover"), ("dem", "Copernicus DEM"))))
    if not use["dem"]:
        sys.exit("The elevation source could not be reached; check the internet connection (or HTTPS_PROXY).")
    cache = adc.Cache(out / "cache")
    local = threading.local()

    def measure(i):
        if not hasattr(local, "r"):
            local.r = {"wc": adc.TileReader("WorldCover", adc.WORLDCOVER_URL, adc.worldcover_tiles, 0),
                       "dem": adc.TileReader("Copernicus DEM", adc.DEM_URL, adc.dem_tiles, -9999)}
        x, y = sites.at[i, "x"], sites.at[i, "y"]
        g_ll = adc.ll(Point(x, y).buffer(args.radius, 8))
        rec = {}
        for k, fn in (("wc", adc.landcover_metrics), ("dem", adc.dem_metrics)):
            if not use[k]:
                continue
            key = cache.key(k, round(x), round(y), args.radius)
            hit = cache.get(k, key)
            if hit is None:
                try:
                    hit = fn(local.r[k], g_ll)
                    cache.put(k, key, hit)
                except Exception as ex:
                    hit = {f"{k}_error": f"{type(ex).__name__}: {str(ex)[:60]}"}
            rec.update(hit)
        return i, rec

    recs = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for j, (i, rec) in enumerate(pool.map(measure, sites.index)):
            recs[i] = rec
            if (j + 1) % 50 == 0 or j + 1 == len(sites):
                print(f"  measured {j + 1}/{len(sites)} circles  [{time.time() - t0:.0f}s]", flush=True)
    sites = sites.join(pd.DataFrame.from_dict(recs, orient="index"))
    n_err = int(sites["dem_relief_m"].isna().sum()) if "dem_relief_m" in sites else len(sites)
    if n_err:
        print(f"  {n_err} circles without elevation (left out of the comparison)")
    sites.drop(columns=["row", "col"]).to_csv(out / "terrain_sites.csv", index=False)

    # ---- compare
    lines, stats = [], {}
    d = sites[sites["dem_relief_m"].notna()].copy()
    d["hilly"] = d["dem_relief_m"] >= HILLY_M
    order = ["gold label", "belt background", "top 1% ground"]
    lines.append(f"{'group':18}{'n':>5}{'median relief m':>17}{'hilly share':>13}{'scrub/bare':>12}{'cropland':>10}")
    for grp in order:
        g = d[d["group"] == grp]
        if g.empty:
            continue
        stats[grp] = {"n": len(g), "relief": float(g["dem_relief_m"].median()), "hilly": float(g["hilly"].mean()),
                      "natural": float(g["wc_natural"].median()) if "wc_natural" in g else np.nan,
                      "crop": float(g["wc_crop"].median()) if "wc_crop" in g else np.nan}
        s = stats[grp]
        lines.append(f"{grp:18}{s['n']:>5}{s['relief']:>17.0f}{s['hilly']:>13.0%}{s['natural']:>12.0%}{s['crop']:>10.0%}")

    def mwu(a, b_):
        x1 = d.loc[d["group"] == a, "dem_relief_m"]
        x2 = d.loc[d["group"] == b_, "dem_relief_m"]
        if len(x1) < 5 or len(x2) < 5:
            return np.nan
        return float(mannwhitneyu(x1, x2, alternative="two-sided").pvalue)

    p_lab = mwu("gold label", "belt background")
    p_top = mwu("top 1% ground", "belt background")
    p_top_lab = mwu("top 1% ground", "gold label")
    bgd = d[d["group"] == "belt background"]
    rho = {}
    if len(bgd) >= 20:
        rho["model_pct~relief"] = float(spearmanr(bgd["model_pct"], bgd["dem_relief_m"], nan_policy="omit").statistic)
        if "wc_natural" in bgd:
            rho["model_pct~scrub/bare"] = float(spearmanr(bgd["model_pct"], bgd["wc_natural"], nan_policy="omit").statistic)
        for k in rad:
            rho[f"{k}~relief"] = float(spearmanr(bgd[k], bgd["dem_relief_m"], nan_policy="omit").statistic)
    lines.append("")
    lines.append(f"relief, gold labels vs background:   p = {p_lab:.3f} (Mann-Whitney)")
    lines.append(f"relief, top 1% vs background:        p = {p_top:.3f}")
    lines.append(f"relief, top 1% vs gold labels:       p = {p_top_lab:.3f}")
    lines.append("rank correlations on belt background (Spearman):")
    for k, v in rho.items():
        lines.append(f"  {k:24} {v:+.2f}")

    # ---- verdict
    lab_s, bg_s, top_s = (stats.get(k, {}) for k in order)
    lines.append("")
    verdict = []
    if lab_s and bg_s:
        lab_hillier = p_lab < 0.05 and lab_s["relief"] > bg_s["relief"]
        top_hillier_lab = p_top_lab < 0.05 and top_s and top_s["relief"] > lab_s["relief"]
        top_hillier_bg = p_top < 0.05 and top_s and top_s["relief"] > bg_s["relief"]
        if lab_hillier:
            verdict.append(f"DETECTION BIAS: gold labels sit on hillier ground than typical belt ground "
                           f"({lab_s['hilly']:.0%} vs {bg_s['hilly']:.0%} hilly). The model can learn 'exposed rock'. "
                           f"V2: sample background matched to the labels' terrain.")
        if top_hillier_lab:
            verdict.append(f"MODEL FAVOURS HILLS beyond the labels ({top_s['hilly']:.0%} of top 1% hilly vs "
                           f"{lab_s['hilly']:.0%} of labels): likely a radiometric/terrain artefact.")
        if not verdict and top_hillier_bg:
            verdict.append("Top ground is hillier than background, but no more than the gold labels: consistent "
                           "with geology (resistant greenstone ridges), no correction needed now.")
        if not verdict:
            verdict.append("No terrain bias found.")
        if abs(rho.get("model_pct~relief", 0)) >= 0.2:
            verdict.append(f"Note: on belt ground the model score tracks relief (rho {rho['model_pct~relief']:+.2f}).")
    else:
        verdict.append("Not enough sites with elevation to compare.")
    lines += verdict
    text = "\n".join(lines)
    (out / "terrain_summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
        data = [d.loc[d["group"] == g, "dem_relief_m"].values for g in order if (d["group"] == g).any()]
        names = [g for g in order if (d["group"] == g).any()]
        try:
            ax[0].boxplot(data, tick_labels=names, showfliers=False)   # matplotlib >= 3.9
        except TypeError:
            ax[0].boxplot(data, labels=names, showfliers=False)
        for i, v in enumerate(data, 1):
            ax[0].scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, s=6, alpha=0.4)
        ax[0].axhline(HILLY_M, ls="--", lw=0.8, color="grey")
        ax[0].set_ylabel("relief in 300 m circle (m)")
        ax[0].set_title("Relief by group")
        if len(bgd):
            ax[1].scatter(bgd["dem_relief_m"], bgd["model_pct"], s=8, alpha=0.5)
            ax[1].set_xlabel("relief (m)")
            ax[1].set_ylabel("model percentile")
            ax[1].set_title(f"Belt background: rho {rho.get('model_pct~relief', np.nan):+.2f}")
        fig.tight_layout()
        fig.savefig(out / "terrain_check.png")
    except Exception as ex:  # plotting is a nice-to-have
        print(f"  (figure skipped: {type(ex).__name__})")
    print(f"\nOutputs in {out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
