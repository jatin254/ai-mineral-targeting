#!/usr/bin/env python3
"""
make_targets.py - turn a V1 prospectivity map into a ranked target list.

Run from the repository folder, after train_v1.py:
    python make_targets.py data/processed --model pg_ens --stability bags
("bags" = the model's own bag maps from train_v1 --bags; run names of other models also work)

Reads   data/processed/models/<model>/prospectivity_pct.tif and run.json, mask_model.tif, stack/*.tif,
        ka_ap_v1.gpkg (gold_labels, exploration_blocks)
Writes  data/processed/models/<model>/targets.gpkg      polygons, EPSG:32643 (for QGIS)
        data/processed/models/<model>/targets.geojson   same, lon/lat (QGIS, geojson.io, Google Earth import)
        data/processed/models/<model>/targets.csv       one row per target
        data/processed/models/<model>/targets.png       map with numbered targets

A target is a connected patch of cells in the top --top % of the model mask (default 1%), at least
--min-km2 in size. For each target:
  status          known (gold inside or within 1 km) / near-known (1-5 km) / new (> 5 km)
  stability       share of the --stability runs (or bags) that also put the target's cells in the top --top %
  pct_min_runs    the target's mean percentile in its worst run; pct_sd_runs = spread across runs.
                  A target whose worst run drops well below the cut-off is model-noise, not geology.
  gsi_gold_blocks GSI exploration blocks with gold that overlap it (report ids and stages)
  evidence        the three features most different from typical belt ground (robust z-scores)
  au_geochem_pct  share of the target with gold geochemistry; low = ranked on geophysics alone
  peak_lon/lat    highest-ranked cell: the first point to check in a large target
  evidence z-scores for gold are large because most gold values sit at the detection limit; read
  them as "strongly elevated", not as exact magnitudes.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from scipy import ndimage
    from shapely.geometry import shape
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

BELT_BUFFER_M = 1000.0
LABELS = {  # short names for the evidence column
    "gc_au_index": "Au pathfinder index", "gc_au": "Au", "gc_as": "As", "gc_sb": "Sb", "gc_bi": "Bi",
    "rad_k": "K", "rad_eth": "eTh", "rad_eu": "eU", "rad_dose": "dose rate", "rad_k_eth": "K/eTh",
    "rad_eu_eth": "eU/eTh", "rad_f": "F-parameter", "rad_heat": "radiogenic heat",
    "grav_ba": "Bouguer", "grav_residual": "gravity residual", "grav_hgm": "gravity gradient",
    "mag_tmi": "magnetic field", "mag_1vd": "magnetic 1VD", "mag_thd": "magnetic gradient",
    "mag_as": "magnetic analytic signal", "mag_tilt": "magnetic tilt",
}


def read(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        return src.read(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a ranked target list from a prospectivity map.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("--model", default="pathf_geophys", help="model run name under models/")
    ap.add_argument("--stability", default="", help="comma-separated run names to measure stability against; "
                                                    "'bags' = this model's bag maps (train_v1 --bags)")
    ap.add_argument("--top", type=float, default=1.0, help="top %% of the model mask that counts as target ground")
    ap.add_argument("--min-km2", type=float, default=0.25, help="smallest target kept")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    proc = Path(args.processed)
    mdir = proc / "models" / args.model
    spec = json.loads((proc / "grid_spec.json").read_text())
    cell = float(spec["cell_m"])
    T = Affine(cell, 0, spec["xmin"], 0, -cell, spec["ymax"])
    run = json.loads((mdir / "run.json").read_text())

    mask = read(proc / "mask_model.tif") == 1
    pct = read(mdir / "prospectivity_pct.tif")
    hot = mask & (pct >= 100 - args.top)

    # stability: share of runs in which each cell is also target ground
    runs = [r.strip() for r in args.stability.split(",") if r.strip()]
    agree = None
    if runs:
        agree = np.zeros(pct.shape, dtype="float32")
        run_pcts = []   # every run's percentile map, for per-target uncertainty
        used = 0
        paths = []
        for r in runs:
            if r == "bags":
                bp = sorted((mdir / "bags").glob("pct_bag*.tif"))
                if not bp:
                    print(f"  stability: no bag maps in {mdir / 'bags'} (train with --bags N), skipped")
                paths += bp
            else:
                p = proc / "models" / r / "prospectivity_pct.tif"
                if p.exists():
                    paths.append(p)
                else:
                    print(f"  stability: {r} not found, skipped")
        for p in paths:
            rp_ = read(p)
            agree += (rp_ >= 100 - args.top).astype("float32")
            run_pcts.append(rp_)
            used += 1
        if used:
            agree = agree / used
            print(f"  stability measured over {used} runs")
        else:
            agree = None

    # connected patches -> polygons
    lab, n = ndimage.label(hot, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    min_cells = int(np.ceil(args.min_km2 / (cell / 1000) ** 2))
    keep_ids = np.nonzero(sizes >= min_cells)[0] + 1
    lab = np.where(np.isin(lab, keep_ids), lab, 0).astype("int32")
    geoms = {}
    for geom, val in shapes(lab, mask=lab > 0, transform=T, connectivity=8):
        geoms.setdefault(int(val), []).append(shape(geom))
    tg = gpd.GeoDataFrame({"patch": list(geoms)}, geometry=[gpd.GeoSeries(g).union_all() for g in geoms.values()],
                          crs=spec["crs"])
    tg["geometry"] = tg.geometry.make_valid().buffer(0)   # raster-traced shapes can touch at corners
    if tg.empty:
        sys.exit("No target patches at this threshold; try a larger --top or smaller --min-km2.")

    # reference sample for evidence: belt-domain cells
    sdist = read(proc / "stack" / "geo_greenstone_sdist.tif")
    belt = mask & (sdist <= BELT_BUFFER_M)
    feats = [f for f in run["features"] if f in LABELS]
    ref, arrays = {}, {}
    for f in feats:
        a = read(proc / "stack" / f"{f}.tif")
        arrays[f] = a
        v = a[belt & np.isfinite(a)]
        med = float(np.median(v)) if v.size else np.nan
        mad = float(np.median(np.abs(v - med)) * 1.4826) if v.size else np.nan
        ref[f] = (med, mad if mad and mad > 0 else (float(np.std(v)) if v.size else np.nan))
    au = read(proc / "stack" / "gc_au.tif")

    rows = []
    for pid in tg["patch"]:
        m = lab == pid
        rr_, cc_ = np.nonzero(m)
        k = int(np.argmax(pct[rr_, cc_]))
        px, py = T * (cc_[k] + 0.5, rr_[k] + 0.5)
        r = {"patch": pid, "peak_x": px, "peak_y": py, "area_km2": round(m.sum() * (cell / 1000) ** 2, 2),
             "mean_pct": round(float(pct[m].mean()), 2), "max_pct": round(float(pct[m].max()), 3),
             "au_geochem_pct": round(100 * float(np.isfinite(au[m]).mean())),
             "stability": round(float(agree[m].mean()), 2) if agree is not None else np.nan}
        if agree is not None and run_pcts:
            per_run = np.array([float(np.nanmean(rp_[m])) for rp_ in run_pcts])
            r["pct_min_runs"] = round(float(per_run.min()), 2)    # worst run: how low can this target fall?
            r["pct_sd_runs"] = round(float(per_run.std()), 3)     # spread of the target's mean percentile
        else:
            r["pct_min_runs"], r["pct_sd_runs"] = np.nan, np.nan
        zs = []
        for f in feats:
            vals = arrays[f][m]
            vals = vals[np.isfinite(vals)]
            med, mad = ref[f]
            if vals.size and np.isfinite(mad) and mad > 0:
                zs.append((f, (float(np.median(vals)) - med) / mad))
        zs.sort(key=lambda t: -abs(t[1]))
        r["evidence"] = "; ".join(f"{LABELS[f]} {z:+.1f}" for f, z in zs[:3])
        rows.append(r)
    tg = tg.merge(pd.DataFrame(rows), on="patch")

    # known gold and GSI exploration
    gold = gpd.read_file(proc / args.gpkg, layer="gold_labels")
    blocks = gpd.read_file(proc / args.gpkg, layer="exploration_blocks")
    gblocks = blocks[blocks["has_gold"].astype(bool)]
    d_gold = tg.geometry.apply(lambda g: float(gold.distance(g).min()))
    tg["gold_inside"] = tg.geometry.apply(lambda g: int(gold.within(g.buffer(1)).sum()))
    tg["dist_known_gold_km"] = (d_gold / 1000).round(1)
    tg["status"] = np.select([d_gold <= 1000, d_gold <= 5000], ["known", "near-known"], "new")

    def block_info(g):
        hits = gblocks[gblocks.intersects(g)]
        if hits.empty:
            return 0, ""
        ids = hits.sort_values("area_km2")["report_id"].astype(str).head(3).tolist()
        stages = ",".join(sorted(set(hits["stage"].astype(str))))
        return len(hits), f"{'; '.join(ids)} [{stages}]"
    info = tg.geometry.apply(block_info)
    tg["gsi_gold_blocks"] = [i[0] for i in info]
    tg["gsi_reports"] = [i[1] for i in info]

    # rank: mean percentile, then stability, then size
    sort_cols = ["mean_pct"] + (["stability"] if agree is not None else []) + ["area_km2"]
    tg = tg.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    tg.insert(0, "target", [f"T{i:03d}" for i in range(1, len(tg) + 1)])
    ll = tg.to_crs("EPSG:4326")
    ll["geometry"] = ll.geometry.make_valid()
    cen = tg.geometry.representative_point().to_crs("EPSG:4326")
    tg["lon"], tg["lat"] = cen.x.round(5), cen.y.round(5)
    peaks = gpd.GeoSeries(gpd.points_from_xy(tg["peak_x"], tg["peak_y"]), crs=spec["crs"]).to_crs("EPSG:4326")
    tg["peak_lon"], tg["peak_lat"] = peaks.x.round(5).values, peaks.y.round(5).values
    for c in ("lon", "lat", "peak_lon", "peak_lat"):
        ll[c] = tg[c]

    cols = ["target", "status", "area_km2", "mean_pct", "max_pct", "stability", "pct_min_runs", "pct_sd_runs",
            "gold_inside", "dist_known_gold_km",
            "gsi_gold_blocks", "gsi_reports", "au_geochem_pct", "evidence", "lon", "lat", "peak_lon", "peak_lat"]
    tg = tg[cols + ["geometry"]]
    for p in ("targets.gpkg", "targets.geojson"):
        (mdir / p).unlink(missing_ok=True)
    tg.to_file(mdir / "targets.gpkg", layer="targets", driver="GPKG")
    ll[cols + ["geometry"]].to_file(mdir / "targets.geojson", driver="GeoJSON")
    tg.drop(columns="geometry").to_csv(mdir / "targets.csv", index=False)

    # map
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 10), dpi=130)
        x0, y1 = spec["xmin"], spec["ymax"]
        ext = [x0, x0 + spec["width"] * cell, y1 - spec["height"] * cell, y1]
        ax.imshow(np.where(mask, pct, np.nan)[::2, ::2], cmap="Greys", vmin=60, vmax=100, extent=ext)
        colours = {"known": "#2b8cbe", "near-known": "#fdae61", "new": "#d7191c"}
        for st, c in colours.items():
            sub = tg[tg.status == st]
            if len(sub):
                sub.boundary.plot(ax=ax, color=c, lw=0.9)
        gold.plot(ax=ax, color="lime", markersize=4)
        for _, r in tg.head(30).iterrows():
            p = r.geometry.representative_point()
            ax.annotate(r.target[1:].lstrip("0"), (p.x, p.y), fontsize=6, color=colours[r.status],
                        xytext=(3, 3), textcoords="offset points", weight="bold")
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=c, lw=2, label=s) for s, c in colours.items()]
                  + [Line2D([], [], marker="o", color="w", markerfacecolor="lime", label="known gold")],
                  loc="lower left", fontsize=7)
        ax.set_axis_off()
        ax.set_title(f"V1 gold targets ({args.model}, top {args.top:g}% of ground; top 30 numbered)", fontsize=10)
        fig.tight_layout()
        fig.savefig(mdir / "targets.png")
    except ImportError:
        pass

    counts = tg.status.value_counts().to_dict()
    print(f"\n{len(tg)} targets ({tg.area_km2.sum():.0f} km2): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    show = tg.head(15).drop(columns="geometry")
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(show[["target", "status", "area_km2", "mean_pct", "stability", "pct_min_runs", "dist_known_gold_km",
                    "gsi_gold_blocks", "au_geochem_pct", "evidence"]].to_string(index=False))
    print(f"\nOutputs in {mdir}: targets.gpkg, targets.geojson, targets.csv, targets.png")


if __name__ == "__main__":
    main()
