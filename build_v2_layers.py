#!/usr/bin/env python3
"""
build_v2_layers.py - V2 input layers from the NGDR map-service downloads: shear zones and soil / stream geochemistry.

Run from the repository folder (after ngdr_wfs_pull.py has saved data\\raw\\ngdr_api\\wfs\\*.geojson):
    python build_v2_layers.py data\\processed

Reads   data\\raw\\ngdr_api\\wfs\\cite_shear_zone_*.geojson         shear zones (1:25,000 and regional sets)
        data\\raw\\ngdr_api\\wfs\\cite_soil_c_horizon_gcs_ngdr.geojson soil samples (C horizon), Au As Sb Bi
        data\\raw\\ngdr_api\\wfs\\cite_stream_sediments_gcs_ngdr.geojson stream-sediment samples, Au As Sb Bi
        data\\processed\\grid_spec.json, mask_model.tif, ka_ap_v1.gpkg (gold_labels, for the check column only)
Writes  data\\processed\\stack\\v2s_dist_shear.tif      m to the nearest shear zone (all shear sets merged)
                                v2s_dens_shear.tif      shear length within 2 km, km per km2
                                v2s_dist_shear_x.tif    m to the nearest point where two shear zones cross or meet
                                v2g_soil_{au,as,sb,bi}.tif, v2g_soil_index.tif   log10 values, inverse-distance
                                                        weighted within 1.5 km of a sample (blank beyond)
                                v2g_ss_{au,as,sb,bi}.tif, v2g_ss_index.tif       the same for stream sediments, 2.5 km
        data\\processed\\stack\\manifest_v2.csv         coverage and where known gold falls on each layer
The index layers are the mean of robust z-scores (median / IQR) of the four log pathfinders, as gc_au_index.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from rasterio import features
    from rasterio.transform import Affine
    from scipy import ndimage, signal
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    from shapely.strtree import STRtree
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

SHEAR_FILES = ["cite_shear_zone_25k_ngdr", "cite_shear_zone_hardrock_stm25k_ngdr", "cite_shear_zone_gcs_ngdr",
               "cite_shear_zone_tectonic_ngdr", "cite_shear_zone_mata_ngdr_gcs"]
GEOCHEM = {"soil": ("cite_soil_c_horizon_gcs_ngdr", 1500.0), "ss": ("cite_stream_sediments_gcs_ngdr", 2500.0)}
ELEMENTS = {"au": ["au"], "as": ["as_", "as"], "sb": ["sb"], "bi": ["bi"]}
DENS_R_M = 2000.0


def load_grid(proc: Path):
    spec = json.loads((proc / "grid_spec.json").read_text())
    T = Affine.from_gdal(*spec["transform_gdal"])
    shape = (spec["height"], spec["width"])
    with rasterio.open(proc / "mask_model.tif") as src:
        mask = src.read(1) == 1
    return spec, T, shape, mask


def write(path: Path, arr: np.ndarray, spec: dict, T, mask: np.ndarray) -> None:
    out = np.where(mask, arr, np.nan).astype("float32")
    prof = dict(driver="GTiff", width=spec["width"], height=spec["height"], count=1, dtype="float32", crs=spec["crs"],
                transform=T, nodata=np.nan, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def read_layer(wfs: Path, stem: str, crs) -> gpd.GeoDataFrame | None:
    p = wfs / f"{stem}.geojson"
    if not p.exists():
        return None
    g = gpd.read_file(p)
    if g.empty:
        return None
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    g = g[g.geometry.notna()]
    g = g[~g.geometry.is_empty]
    return g.to_crs(crs)


def shear_layers(wfs: Path, spec, T, shape, mask, out: Path, log: list) -> None:
    parts = []
    for stem in SHEAR_FILES:
        g = read_layer(wfs, stem, spec["crs"])
        if g is not None:
            g = g[g.geom_type.isin(["LineString", "MultiLineString"])]
            parts.append(g.geometry)
            log.append(f"  {stem}: {len(g)} shear segments")
        else:
            log.append(f"  {stem}: not found (skipped)")
    if not parts:
        log.append("  no shear-zone layers found: v2s_ layers not written")
        return
    geoms = [geom for s in parts for geom in s]
    line = features.rasterize(((gm, 1) for gm in geoms), out_shape=shape, transform=T, all_touched=True, dtype="uint8")
    cell = abs(T.a)
    dist = ndimage.distance_transform_edt(line == 0) * cell
    write(out / "v2s_dist_shear.tif", dist, spec, T, mask)
    # length density: line cells within a disk, each cell ~ one cell length of line
    r = int(DENS_R_M / cell)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r * r).astype("float32")
    cnt = signal.fftconvolve(line.astype("float32"), disk, mode="same")
    dens = np.clip(cnt, 0, None) * (cell / 1000.0) / (disk.sum() * (cell / 1000.0) ** 2)
    write(out / "v2s_dens_shear.tif", dens, spec, T, mask)
    # crossings and junctions between different shear segments
    # crossings: two different shears meeting at a point. Not counted: overlaps (the same shear drawn in two
    # layers intersects along a line) and two pieces of one shear joined end to end. One crossing per 200 m.
    def ends(gm):
        parts = [gm] if gm.geom_type == "LineString" else list(gm.geoms)
        return [Point(q.coords[k]) for q in parts for k in (0, -1)]
    tree = STRtree(geoms)
    pts, n_overlap, n_joined = [], 0, 0
    for i, gm in enumerate(geoms):
        ei = ends(gm)
        for j in tree.query(gm):
            if j <= i:
                continue
            x = gm.intersection(geoms[j])
            if x.is_empty:
                continue
            if isinstance(x, Point):
                cand = [x]
            elif hasattr(x, "geoms"):
                cand = [q for q in x.geoms if q.geom_type == "Point"]
                n_overlap += int(any(q.geom_type != "Point" for q in x.geoms))
            else:
                n_overlap += 1
                continue
            ej = ends(geoms[j])
            for q in cand:
                if any(q.distance(e) < 1.0 for e in ei) and any(q.distance(e) < 1.0 for e in ej):
                    n_joined += 1
                    continue
                pts.append(q)
    cells200 = {(round(q.x / 200), round(q.y / 200)) for q in pts}
    pts = [Point(cx * 200, cy * 200) for cx, cy in cells200]
    log.append(f"  shear crossings: {len(pts)} distinct (left out: {n_overlap} overlaps of the same shear drawn twice, "
               f"{n_joined} end-to-end joins)")
    if pts:
        xr = features.rasterize(((p.buffer(cell / 2), 1) for p in pts), out_shape=shape, transform=T, all_touched=True,
                                dtype="uint8")
        write(out / "v2s_dist_shear_x.tif", ndimage.distance_transform_edt(xr == 0) * cell, spec, T, mask)


def idw(points_xy: np.ndarray, vals: np.ndarray, cells_xy: np.ndarray, radius: float, k: int = 8) -> np.ndarray:
    tree = cKDTree(points_xy)
    out = np.full(len(cells_xy), np.nan, dtype="float64")
    for s in range(0, len(cells_xy), 400_000):
        d, ix = tree.query(cells_xy[s:s + 400_000], k=min(k, len(points_xy)), distance_upper_bound=radius)
        if d.ndim == 1:
            d, ix = d[:, None], ix[:, None]
        ok = np.isfinite(d)
        w = np.where(ok, 1.0 / np.maximum(d, 50.0) ** 2, 0.0)
        v = np.where(ok, vals[np.minimum(ix, len(vals) - 1)], 0.0)
        ws = w.sum(1)
        out[s:s + 400_000] = np.where(ws > 0, (w * v).sum(1) / np.where(ws > 0, ws, 1), np.nan)
    return out


def geochem_layers(wfs: Path, spec, T, shape, mask, out: Path, log: list) -> None:
    rows, cols = np.nonzero(mask)
    cx = T.c + (cols + 0.5) * T.a
    cy = T.f + (rows + 0.5) * T.e
    cells_xy = np.c_[cx, cy]
    for key, (stem, radius) in GEOCHEM.items():
        g = read_layer(wfs, stem, spec["crs"])
        if g is None:
            log.append(f"  {stem}: not found (skipped)")
            continue
        g = g[g.geom_type.isin(["Point", "MultiPoint"])]
        g = g.set_geometry(g.geometry.representative_point())
        xy = np.c_[g.geometry.x.to_numpy(), g.geometry.y.to_numpy()]
        n_loc = len({(round(x_), round(y_)) for x_, y_ in xy})
        log.append(f"  {stem}: {len(g)} samples at {n_loc} distinct positions")
        zs = []
        for el, cands in ELEMENTS.items():
            col = next((c for c in cands if c in g.columns), None)
            if col is None:
                log.append(f"  {stem}: no {el} column")
                continue
            v = pd.to_numeric(g[col].astype(str).str.replace("<", "-", regex=False).str.strip(), errors="coerce").to_numpy(
                dtype="float64")                                                 # '<5' -> -5: below detection
            pos = v[np.isfinite(v) & (v > 0)]
            n_low = int((np.isfinite(v) & (v <= 0)).sum())
            vals_, counts_ = np.unique(pos, return_counts=True) if len(pos) else (np.array([]), np.array([0]))
            top_share = counts_.max() / max(len(pos), 1)
            log.append(f"    {el}: {len(v)} rows, {len(pos)} positive ({len(vals_)} distinct values), {n_low} below "
                       f"detection, {int((~np.isfinite(v)).sum())} blank; positive min / median / max "
                       + (f"{pos.min():g} / {np.median(pos):g} / {pos.max():g}" if len(pos) else "-")
                       + f"; most common value covers {top_share:.0%}")
            if len(pos) < 20 or len(vals_) < 20 or top_share > 0.6:
                log.append(f"    {el}: SKIPPED - too few distinct values to make a layer (check the source table)")
                continue
            v = np.where(np.isfinite(v) & (v <= 0), pos.min() / 2, v)        # below detection -> half the lowest value
            ok = np.isfinite(v)
            lv = np.log10(v[ok])
            grid = np.full(shape, np.nan, dtype="float32")
            grid[rows, cols] = idw(xy[ok], lv, cells_xy, radius)
            write(out / f"v2g_{key}_{el}.tif", grid, spec, T, mask)
            med, iqr = np.nanmedian(lv), np.subtract(*np.nanpercentile(lv, [75, 25]))
            zs.append((grid - med) / (iqr if iqr > 0 else 1.0))
            log.append(f"  {stem}: {el} from {int(ok.sum())} samples, "
                       f"{np.isfinite(grid[rows, cols]).mean():.0%} of the mask within {radius / 1000:g} km")
        if zs:
            with warnings.catch_warnings():               # cells no sample reaches stay blank
                warnings.simplefilter("ignore", RuntimeWarning)
                write(out / f"v2g_{key}_index.tif", np.nanmean(np.stack(zs), axis=0), spec, T, mask)


def manifest(proc: Path, out: Path, T, mask) -> list[dict]:
    try:
        lab = gpd.read_file(proc / "ka_ap_v1.gpkg", layer="gold_labels")
        lr, lc = rasterio.transform.rowcol(T, lab.geometry.x.values, lab.geometry.y.values)
        lr, lc = np.asarray(lr), np.asarray(lc)
    except Exception:
        lr = lc = None
    rows = []
    for p in sorted(out.glob("v2*.tif")):
        with rasterio.open(p) as src:
            a = src.read(1)
        v = a[mask]
        fin = np.isfinite(v)
        r = {"feature": p.stem, "coverage_pct_of_mask": round(100 * fin.mean(), 1),
             "p50": round(float(np.nanmedian(v)), 4) if fin.any() else ""}
        if lr is not None and fin.any():
            ok = (lr >= 0) & (lr < a.shape[0]) & (lc >= 0) & (lc < a.shape[1])
            g = a[lr[ok], lc[ok]]
            g = g[np.isfinite(g)]
            r["gold_labels_covered"] = int(len(g))
            if len(g):
                r["median_at_gold"] = round(float(np.median(g)), 4)
                r["gold_percentile"] = int(round(100 * np.mean(v[fin] <= np.median(g))))
        rows.append(r)
    with open(out / "manifest_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "coverage_pct_of_mask", "p50", "gold_labels_covered", "median_at_gold",
                                          "gold_percentile"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="V2 layers: shear zones and soil / stream geochemistry from NGDR downloads.")
    ap.add_argument("processed")
    ap.add_argument("--wfs", default="data/raw/ngdr_api/wfs", help="folder of the NGDR map-service downloads")
    args = ap.parse_args()
    proc, wfs = Path(args.processed), Path(args.wfs)
    if not wfs.exists():
        sys.exit(f"{wfs} not found: run ngdr_wfs_pull.py first")
    spec, T, shape, mask = load_grid(proc)
    out = proc / "stack"
    for old in list(out.glob("v2s_*.tif")) + list(out.glob("v2g_*.tif")):   # no stale layer survives a rebuild
        old.unlink()
    log = ["Shear zones:"]
    shear_layers(wfs, spec, T, shape, mask, out, log)
    log.append("Geochemistry:")
    geochem_layers(wfs, spec, T, shape, mask, out, log)
    print("\n".join(log))
    rows = manifest(proc, out, T, mask)
    print("\nLayer                 coverage   median   labels  at gold   gold percentile (low = gold sits at low values)")
    for r in rows:
        print(f"  {r['feature']:20} {r['coverage_pct_of_mask']:6}%  {str(r['p50']):>9}  {str(r.get('gold_labels_covered', '')):>6}  "
              f"{str(r.get('median_at_gold', '')):>9}  {r.get('gold_percentile', '')}")
    print(f"\nWritten {out}\\v2*.tif and manifest_v2.csv")


if __name__ == "__main__":
    main()
