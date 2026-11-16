#!/usr/bin/env python3
"""
build_terrain_layers.py - terrain and land-cover layers on the model grid, for measuring and
correcting the terrain (exposure) bias. They are NOT model features.

Why: terrain_bias_check.py found the gold labels on hillier, scrub-covered ground than typical belt
ground (69% vs 21% hilly; 1% vs 61% cropland). Old workings were found where rock is exposed, so the
model can learn "exposed rock" instead of "gold". These layers let train_v1.py
  * measure capture within the same terrain (does the model beat random once terrain is held equal?)
  * sample background that matches the labels' terrain (--match-terrain), so terrain alone cannot
    separate gold from background ("target-group background", Phillips et al. 2009, Ecol. Appl. 19:181).

Run from the repository folder (needs internet once, about 2-6 minutes):
    python build_terrain_layers.py data/processed

Reads   Copernicus DEM GLO-30 and ESA WorldCover 2021 (same open sources as the desk check), read at
        reduced resolution straight from their cloud-optimised files
Writes  data/processed/terrain/
          dem.tif      elevation, m (100 m cells)
          relief.tif   max - min elevation within 300 m, m
          scrub.tif    share of shrub + grass + bare ground in the cell (0-1), from WorldCover
          crop.tif     share of cropland;  tree.tif, built.tif, water.tif likewise
          terrain.png  quicklook
          README.txt
Credits: Copernicus DEM GLO-30 (ESA); ESA WorldCover 2021 v200 (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_desk_check as adc  # noqa: E402  (sets the GDAL network options before rasterio loads)

import numpy as np  # noqa: E402

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    from scipy import ndimage
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

DEM_STEP_DEG = 1 / 1200      # ~90 m: read the 30 m DEM at a third of its resolution
WC_STEP_DEG = 1 / 3000       # ~37 m: read the 10 m WorldCover at a quarter of its resolution
RELIEF_RADIUS_M = 300.0
CLASSES = {"scrub": (20, 30, 60), "crop": (40,), "tree": (10,), "built": (50,), "water": (80, 90)}


def read_decimated(url: str, bounds, step_deg: float, resampling):
    """Read the part of a lon/lat tile inside `bounds` at roughly `step_deg` pixels (uses overviews)."""
    with rasterio.open(adc.gdal_path(url)) as ds:
        b = (max(bounds[0], ds.bounds.left), max(bounds[1], ds.bounds.bottom),
             min(bounds[2], ds.bounds.right), min(bounds[3], ds.bounds.top))
        if b[0] >= b[2] or b[1] >= b[3]:
            return None
        win = window_from_bounds(*b, transform=ds.transform)
        w = max(1, int(round((b[2] - b[0]) / step_deg)))
        h = max(1, int(round((b[3] - b[1]) / step_deg)))
        arr = ds.read(1, window=win, out_shape=(h, w), resampling=resampling)
        tr = Affine((b[2] - b[0]) / w, 0, b[0], 0, -(b[3] - b[1]) / h, b[3])
        return arr, tr, ds.nodata


def main() -> None:
    ap = argparse.ArgumentParser(description="Terrain and land-cover layers on the model grid.")
    ap.add_argument("processed", help="data/processed")
    args = ap.parse_args()
    proc = Path(args.processed)
    out = proc / "terrain"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = json.loads((proc / "grid_spec.json").read_text())
    cell = float(spec["cell_m"])
    H, W = int(spec["height"]), int(spec["width"])
    T = Affine(cell, 0, spec["xmin"], 0, -cell, spec["ymax"])
    crs = spec["crs"]
    ll = transform_bounds(crs, "EPSG:4326", spec["xmin"], spec["ymin"], spec["xmax"], spec["ymax"])
    ll = (ll[0] - 0.02, ll[1] - 0.02, ll[2] + 0.02, ll[3] + 0.02)
    prof = dict(driver="GTiff", width=W, height=H, count=1, dtype="float32", crs=crs, transform=T,
                nodata=np.nan, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(proc / "mask_model.tif") as src:
        mask = src.read(1) == 1

    # ---- elevation and relief
    if not any(adc.reachable(adc.DEM_URL.format(tile=t)) for t in adc.dem_tiles(ll)):
        sys.exit("Copernicus DEM could not be reached; check the internet connection (or HTTPS_PROXY).")
    dem = np.full((H, W), np.nan, dtype="float32")
    for tile in adc.dem_tiles(ll):
        try:
            got = read_decimated(adc.DEM_URL.format(tile=tile), ll, DEM_STEP_DEG, Resampling.average)
        except Exception as ex:
            print(f"  DEM {tile}: not available ({type(ex).__name__}), skipped")
            continue
        if got is None:
            continue
        arr, tr, nd = got
        arr = arr.astype("float32")
        if nd is not None:
            arr[arr == nd] = np.nan
        arr[arr < -1000] = np.nan
        reproject(arr, dem, src_transform=tr, src_crs="EPSG:4326", dst_transform=T, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, init_dest_nodata=False, resampling=Resampling.bilinear)
        print(f"  DEM {tile} read  [{time.time() - t0:.0f}s]", flush=True)
    r = int(math.ceil(RELIEF_RADIUS_M / cell))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    foot = (xx ** 2 + yy ** 2) * cell ** 2 <= RELIEF_RADIUS_M ** 2
    valid = np.isfinite(dem)
    hi = ndimage.maximum_filter(np.where(valid, dem, -np.inf), footprint=foot)
    lo = ndimage.minimum_filter(np.where(valid, dem, np.inf), footprint=foot)
    relief = np.where(valid & np.isfinite(hi) & np.isfinite(lo), hi - lo, np.nan).astype("float32")
    for name, a in (("dem", dem), ("relief", relief)):
        with rasterio.open(out / f"{name}.tif", "w", **prof) as dst:
            dst.write(a, 1)

    # ---- land-cover shares
    fr = {k: np.full((H, W), np.nan, dtype="float32") for k in CLASSES}
    wc_ok = any(adc.reachable(adc.WORLDCOVER_URL.format(tile=t)) for t in adc.worldcover_tiles(ll))
    if wc_ok:
        for tile in adc.worldcover_tiles(ll):
            try:
                got = read_decimated(adc.WORLDCOVER_URL.format(tile=tile), ll, WC_STEP_DEG, Resampling.nearest)
            except Exception as ex:
                print(f"  WorldCover {tile}: not available ({type(ex).__name__}), skipped")
                continue
            if got is None:
                continue
            arr, tr, _ = got
            have = arr > 0
            for k, codes in CLASSES.items():
                ind = np.where(have, np.isin(arr, codes).astype("float32"), np.nan)
                reproject(ind, fr[k], src_transform=tr, src_crs="EPSG:4326", dst_transform=T, dst_crs=crs,
                          src_nodata=np.nan, dst_nodata=np.nan, init_dest_nodata=False, resampling=Resampling.average)
            print(f"  WorldCover {tile} read  [{time.time() - t0:.0f}s]", flush=True)
        for k, a in fr.items():
            with rasterio.open(out / f"{k}.tif", "w", **prof) as dst:
                dst.write(a, 1)
    else:
        print("  WorldCover could not be reached: land-cover shares not written")

    # ---- summary, labels vs belt ground on the full grid
    msg = [f"relief within {RELIEF_RADIUS_M:.0f} m: median {np.nanmedian(relief[mask]):.0f} m inside the model mask"]
    try:
        import geopandas as gpd
        from rasterio.transform import rowcol
        lab = gpd.read_file(proc / "ka_ap_v1.gpkg", layer="gold_labels").to_crs(crs)
        lr, lc = rowcol(T, lab.geometry.x.values, lab.geometry.y.values)
        lr, lc = np.clip(np.asarray(lr), 0, H - 1), np.clip(np.asarray(lc), 0, W - 1)
        with rasterio.open(proc / "stack" / "geo_greenstone_sdist.tif") as src:
            belt = mask & (src.read(1) <= 1000)
        msg.append(f"{'':18}{'median relief m':>16}{'scrub share':>13}{'cropland':>10}")
        for name, sel_r, sel_c in (("gold labels", lr, lc), ("belt ground", *np.nonzero(belt))):
            rel = np.nanmedian(relief[sel_r, sel_c])
            sc = np.nanmedian(fr["scrub"][sel_r, sel_c]) if wc_ok else np.nan
            cr = np.nanmedian(fr["crop"][sel_r, sel_c]) if wc_ok else np.nan
            msg.append(f"{name:18}{rel:>16.0f}{sc:>13.0%}{cr:>10.0%}")
    except Exception as ex:  # summary only
        msg.append(f"(label summary skipped: {type(ex).__name__})")
    print("\n".join(msg))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 6), dpi=110)
        for a, (title, arr, cmap, vmax) in zip(ax, (("relief within 300 m (m)", relief, "terrain", 100),
                                                    ("scrub / bare share", fr["scrub"], "YlOrBr", 1))):
            im = a.imshow(np.where(mask, arr, np.nan), cmap=cmap, vmin=0, vmax=vmax)
            a.set_title(title)
            a.set_axis_off()
            fig.colorbar(im, ax=a, shrink=0.7)
        fig.tight_layout()
        fig.savefig(out / "terrain.png")
    except Exception:
        pass
    (out / "README.txt").write_text(__doc__.strip() + "\n\n" + "\n".join(msg) + "\n", encoding="utf-8")
    print(f"Outputs in {out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
