#!/usr/bin/env python3
"""
build_geochem_grid.py - put the NGCM stream-sediment geochemistry on the 100 m grid.

Run from the repository folder, after build_vector_features.py:
    python build_geochem_grid.py data/processed

Reads   data/processed/ka_ap_v1.gpkg (layer geochem_ngcm), grid_spec.json, mask_model.tif
Writes  data/processed/stack/gc_<element>.tif   log10 concentration, inverse-distance weighted
        data/processed/stack/gc_au_index.tif     gold pathfinder index (mean robust z of Au, As, Sb, Bi)
        data/processed/geochem_coverage.tif     1 where gold/pathfinders were analysed nearby, 0 where not
        data/processed/stack/manifest_geochem.csv

Method: each cell takes the inverse-distance-squared mean of up to 8 analysed samples within 4 km,
in log10 space. Samples are ~2 km apart, so these are smooth regional layers, not 100 m detail.
Cells with no analysed sample within 4 km stay NaN; for gold and its pathfinders that covers the
20 map sheets that were never analysed. Zeros remaining after cleaning are treated as missing.
The coverage raster is deliberately kept out of stack/ so the model cannot learn "where GSI analysed".
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
    import rasterio
    from rasterio.transform import Affine, rowcol
    from scipy.spatial import cKDTree
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

K, RADIUS, POWER = 8, 4000.0, 2.0
ELEMENTS = {  # column -> output name; Mo dropped (>80% at detection limit), W kept but weak
    "Au_ppb": "gc_au", "As_ppm": "gc_as", "Sb_ppm": "gc_sb", "Bi_ppm": "gc_bi", "Ag_ppb": "gc_ag",
    "Hg_ppb": "gc_hg", "W_ppm": "gc_w", "Cu_ppm": "gc_cu", "Pb_ppm": "gc_pb", "Zn_ppm": "gc_zn",
    "Ni_ppm": "gc_ni", "Cr_ppm": "gc_cr", "MgO_%": "gc_mgo", "K2O_%": "gc_k2o",
}
INDEX_ELEMENTS = ["Au_ppb", "As_ppm", "Sb_ppm", "Bi_ppm"]
INDEX_MIN = 3   # a sample needs at least 3 of the 4 to get an index value
CHUNK = 400_000


class Grid:
    def __init__(self, spec: dict):
        self.crs = spec["crs"]
        self.cell = float(spec["cell_m"])
        self.width, self.height = int(spec["width"]), int(spec["height"])
        self.shape = (self.height, self.width)
        self.transform = Affine(self.cell, 0, spec["xmin"], 0, -self.cell, spec["ymax"])

    def profile(self, dtype="float32", nodata=np.nan) -> dict:
        return dict(driver="GTiff", width=self.width, height=self.height, count=1, dtype=dtype,
                    crs=self.crs, transform=self.transform, nodata=nodata, compress="deflate",
                    tiled=True, blockxsize=256, blockysize=256)


def write(path: Path, a: np.ndarray, grid: Grid, dtype="float32", nodata=np.nan) -> None:
    with rasterio.open(path, "w", **grid.profile(dtype, nodata)) as dst:
        dst.write(a.astype(dtype), 1)


def idw(sample_xy: np.ndarray, values: np.ndarray, cell_xy: np.ndarray) -> np.ndarray:
    tree = cKDTree(sample_xy)
    out = np.full(len(cell_xy), np.nan, dtype="float32")
    for s in range(0, len(cell_xy), CHUNK):
        d, i = tree.query(cell_xy[s:s + CHUNK], k=K, distance_upper_bound=RADIUS, workers=-1)
        valid = np.isfinite(d)
        w = np.where(valid, 1.0 / np.maximum(d, 1.0) ** POWER, 0.0)
        v = np.where(valid, values[np.minimum(i, len(values) - 1)], 0.0)
        wsum = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + CHUNK] = np.where(wsum > 0, (w * v).sum(axis=1) / wsum, np.nan)
    return out


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 0 else np.nanstd(x))


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid NGCM geochemistry onto the model grid.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    proc = Path(args.processed)
    grid = Grid(json.loads((proc / "grid_spec.json").read_text()))
    stack = proc / "stack"
    t0 = time.time()

    with rasterio.open(proc / "mask_model.tif") as src:
        mask = src.read(1) == 1
    rows, cols = np.nonzero(mask)
    cx = grid.transform.c + (cols + 0.5) * grid.cell
    cy = grid.transform.f - (rows + 0.5) * grid.cell
    cell_xy = np.c_[cx, cy]

    g = gpd.read_file(proc / args.gpkg, layer="geochem_ngcm").to_crs(grid.crs)
    sxy = np.c_[g.geometry.x.values, g.geometry.y.values]

    logs = {}
    manifest = []
    for col, name in ELEMENTS.items():
        v = g[col].to_numpy(dtype="float64", copy=True)
        v[v <= 0] = np.nan                       # remaining zeros: treat as missing
        lv = np.log10(v)
        logs[col] = lv
        ok = np.isfinite(lv)
        vals = idw(sxy[ok], lv[ok], cell_xy)
        out = np.full(grid.shape, np.nan, dtype="float32")
        out[rows, cols] = vals
        write(stack / f"{name}.tif", out, grid)
        manifest.append({"feature": name, "source_column": col, "units": f"log10({col.split('_')[-1]})",
                         "samples_used": int(ok.sum()), "coverage_pct_of_mask": round(100 * np.isfinite(vals).mean(), 1),
                         "p50_log": float(np.nanmedian(vals))})
        print(f"  {name:7} <- {col:7} samples={ok.sum():5}  coverage={np.isfinite(vals).mean():6.1%}  [{time.time() - t0:.0f}s]", flush=True)

    # gold pathfinder index: mean robust z of log Au, As, Sb, Bi per sample (>= 3 present)
    Z = np.column_stack([robust_z(logs[c]) for c in INDEX_ELEMENTS])
    n_ok = np.isfinite(Z).sum(axis=1)
    idx = np.where(n_ok >= INDEX_MIN, np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=1), np.nan)
    ok = np.isfinite(idx)
    vals = idw(sxy[ok], idx[ok], cell_xy)
    out = np.full(grid.shape, np.nan, dtype="float32")
    out[rows, cols] = vals
    write(stack / "gc_au_index.tif", out, grid)
    manifest.append({"feature": "gc_au_index", "source_column": "+".join(INDEX_ELEMENTS), "units": "robust z",
                     "samples_used": int(ok.sum()), "coverage_pct_of_mask": round(100 * np.isfinite(vals).mean(), 1),
                     "p50_log": float(np.nanmedian(vals))})

    # coverage of gold analyses (reference only; not a model feature)
    cov = np.zeros(grid.shape, dtype="uint8")
    au = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(stack / "gc_au.tif") as src:
        au = src.read(1)
    cov[np.isfinite(au)] = 1
    write(proc / "geochem_coverage.tif", cov, grid, dtype="uint8", nodata=255)

    # gold labels vs gold geochemistry coverage
    lab = gpd.read_file(proc / args.gpkg, layer="gold_labels")
    r, c = rowcol(grid.transform, lab.geometry.x.values, lab.geometry.y.values)
    r, c = np.clip(np.asarray(r), 0, grid.height - 1), np.clip(np.asarray(c), 0, grid.width - 1)
    occ = (lab.label_class == "occurrence").values
    covered = cov[r, c] == 1

    with open(stack / "manifest_geochem.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        for m in manifest:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in m.items()})

    print(f"\nWrote {len(manifest)} geochem layers to {stack}  [{time.time() - t0:.0f}s]")
    print(f"Gold geochem covers {cov[mask].mean():.1%} of the model mask")
    print(f"Gold occurrences with gold geochem: {int((covered & occ).sum())} of {int(occ.sum())}; "
          f"old-working sites: {int((covered & ~occ).sum())} of {int((~occ).sum())}")


if __name__ == "__main__":
    main()
