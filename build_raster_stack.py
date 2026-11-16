#!/usr/bin/env python3
"""
build_raster_stack.py - put the geophysics on the 100 m model grid and derive the V1 raster features.

Run from the repository folder, after prepare_vectors.py:
    python build_raster_stack.py data/raw/aikosh-gsi-ka-ap data/processed

Reads   data/processed/grid_spec.json and the aoi + gold_labels layers of data/processed/ka_ap_v1.gpkg
Writes  data/processed/stack/<feature>.tif   float32, grid from grid_spec.json, EPSG:32643, NaN = no data
        data/processed/stack/manifest.csv    source, method, units, coverage, value range, gold-vs-background
        data/processed/stack/stack_quicklook.png   (if matplotlib is installed)

Features (16):
  Magnetics  mag_tmi, mag_1vd (vertical derivative), mag_thd (horizontal gradient), mag_as (analytic
             signal), mag_tilt. Derivatives use the field upward-continued 100 m to damp noise.
             No reduction-to-pole: unstable at South India's low magnetic latitude.
  Gravity    grav_ba (Bouguer), grav_residual (BA minus 10 km Gaussian regional), grav_hgm (gradient)
  Radiometric rad_k, rad_eth, rad_eu, rad_dose, rad_k_eth (potassic alteration), rad_eu_eth,
             rad_f (K*eU/eTh alteration index), rad_heat (radiogenic heat production only).
             V2: ratios are missing where eTh < 2 ppm (they were unstable there), and all radiometric
             layers are missing over water bodies mapped in the 250K geomorphology (water_classes.csv
             lists the classes used).

Source grids finer than the cell are block-averaged; coarser ones (gravity, 500 m) are bilinear-interpolated.
Geosoft dummy values (~1.7e38) are treated as no data.
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
    import pyogrio
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask, rasterize
    from rasterio.transform import Affine, rowcol
    from rasterio.warp import reproject
    from scipy import ndimage
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

DUMMY = 1e30                  # anything larger in magnitude is a Geosoft dummy
REGIONAL_SIGMA_M = 10_000     # gravity regional: Gaussian low-pass, 10 km
HEAT_DENSITY = 2700.0         # kg/m3, assumed crustal density for radiogenic heat (Rybach 1988)
UPCONT_M = 100.0              # magnetic derivatives use the field upward-continued by one cell
RATIO_MIN_ETH = 2.0           # ppm; eU/eTh, K/eTh and F only computed where eTh >= this

SOURCES = {  # key: (glob under raw/, units)
    "tmi":  ("MAGNETIC/GRIDS/GEOTIFF/*TMI*.tif*", "nT"),
    "k":    ("SPECTROMETRIC/GRIDS/GEOTIFF/K_perc.tif*", "%"),
    "eth":  ("SPECTROMETRIC/GRIDS/GEOTIFF/eTh_ppm.tif*", "ppm"),
    "eu":   ("SPECTROMETRIC/GRIDS/GEOTIFF/eU_ppm.tif*", "ppm"),
    "dose": ("SPECTROMETRIC/GRIDS/GEOTIFF/Dose_rate*.tif*", "nGy/h"),
    "ba":   ("Ground_gravity*/GEOTIFF/*BA*.tif*", "mGal"),
}


# ----------------------------------------------------------------------------- grid + io

class Grid:
    def __init__(self, spec: dict):
        self.crs = spec["crs"]
        self.cell = float(spec["cell_m"])
        self.width, self.height = int(spec["width"]), int(spec["height"])
        self.transform = Affine(self.cell, 0, spec["xmin"], 0, -self.cell, spec["ymax"])

    def empty(self) -> np.ndarray:
        return np.full((self.height, self.width), np.nan, dtype="float32")

    def profile(self) -> dict:
        return dict(driver="GTiff", width=self.width, height=self.height, count=1, dtype="float32",
                    crs=self.crs, transform=self.transform, nodata=np.nan, compress="deflate",
                    predictor=3, tiled=True, blockxsize=256, blockysize=256)


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(p for p in root.glob(pattern) if p.suffix.lower() in (".tif", ".tiff"))
    if not hits:
        sys.exit(f"Not found under {root}: {pattern}")
    return hits[0]


def to_grid(path: Path, grid: Grid) -> tuple[np.ndarray, str]:
    """Read band 1, clean no-data, and resample onto the model grid."""
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        if src.nodata is not None and np.isfinite(src.nodata):
            a[a == np.float32(src.nodata)] = np.nan
        a[~np.isfinite(a) | (np.abs(a) > DUMMY)] = np.nan
        coarser = max(abs(src.res[0]), abs(src.res[1])) > grid.cell
        method = Resampling.bilinear if coarser else Resampling.average
        out = grid.empty()
        reproject(a, out, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
                  resampling=method)
    return out, method.name


def write(path: Path, a: np.ndarray, grid: Grid) -> None:
    with rasterio.open(path, "w", **grid.profile()) as dst:
        dst.write(a.astype("float32"), 1)


# ----------------------------------------------------------------------------- maths

def fill_nearest(a: np.ndarray) -> np.ndarray:
    m = np.isnan(a)
    if not m.any():
        return a.astype("float64")
    idx = ndimage.distance_transform_edt(m, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype("float64")


def upcont_and_dz(f: np.ndarray, cell: float, height: float, pad: float = 0.25):
    """Upward-continue by `height` m and take the first vertical derivative, both by FFT.
    Reflect padding limits edge ringing; the upward continuation damps short-wavelength noise
    that derivatives would otherwise amplify."""
    ny, nx = f.shape
    py, px = int(ny * pad), int(nx * pad)
    p = np.pad(f - f.mean(), ((py, py), (px, px)), mode="reflect")
    ky = 2 * np.pi * np.fft.fftfreq(p.shape[0], d=cell)
    kx = 2 * np.pi * np.fft.rfftfreq(p.shape[1], d=cell)
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    F = np.fft.rfft2(p) * np.exp(-k * height)
    up = np.fft.irfft2(F, s=p.shape)[py:py + ny, px:px + nx]
    dz = np.fft.irfft2(F * k, s=p.shape)[py:py + ny, px:px + nx]
    return up, dz


def magnetic_features(tmi: np.ndarray, cell: float) -> dict[str, np.ndarray]:
    up, dz = upcont_and_dz(fill_nearest(tmi), cell, height=UPCONT_M)
    dy, dx = np.gradient(up, cell)
    thd = np.hypot(dx, dy)
    return {
        "mag_1vd": dz,                                   # nT/m
        "mag_thd": thd,                                  # nT/m
        "mag_as": np.sqrt(dx ** 2 + dy ** 2 + dz ** 2),  # nT/m
        "mag_tilt": np.degrees(np.arctan2(dz, thd)),     # degrees, -90..90
    }


def gravity_features(ba: np.ndarray, cell: float) -> dict[str, np.ndarray]:
    f = fill_nearest(ba)
    # remove a best-fit plane first so the Gaussian regional doesn't bend at the grid edges
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(f.size)]
    sub = slice(None, None, 7)  # fit on a subsample; a plane needs few points
    coef, *_ = np.linalg.lstsq(A[sub], f.ravel()[sub], rcond=None)
    plane = (A @ coef).reshape(f.shape)
    d = f - plane
    regional = plane + ndimage.gaussian_filter(d, sigma=REGIONAL_SIGMA_M / cell, mode="reflect")
    gy, gx = np.gradient(f, cell)
    return {"grav_residual": f - regional, "grav_hgm": np.hypot(gx, gy) * 1000.0}  # mGal, mGal/km


def radiometric_features(k, eth, eu) -> dict[str, np.ndarray]:
    k0, th0, u0 = np.clip(k, 0, None), np.clip(eth, 0, None), np.clip(eu, 0, None)
    kf, uf = np.clip(k, 0.1, None), np.clip(eu, 0.1, None)
    # Ratios over eTh are only trusted where eTh is well above its noise; below RATIO_MIN_ETH they
    # explode (z-scores of +10 to +22 in V1) and are set to missing instead.
    reliable = np.isfinite(eth) & (eth >= RATIO_MIN_ETH)
    thf = np.where(reliable, eth, np.nan)
    return {
        "rad_k_eth": kf / thf,                    # % per ppm; high = potassic alteration
        "rad_eu_eth": uf / thf,
        "rad_f": kf * uf / thf,                   # F-parameter alteration index
        # radiogenic HEAT production (Rybach 1988)
        "rad_heat": 1e-5 * HEAT_DENSITY * (9.52 * u0 + 2.56 * th0 + 3.48 * k0),  # microW/m3
    }


WATER_KEYWORDS = ("water", "reservoir", "tank", "lake", "pond")


def water_mask(raw: Path, grid: "Grid", out_dir: Path) -> np.ndarray | None:
    """Rasterise water-body classes from the GSI 250K geomorphology map (if present).
    Every class and whether it was treated as water is written to water_classes.csv for review."""
    hits = sorted(raw.glob("Geomorphology*/*.shp"))
    if not hits:
        print("  no geomorphology layer found; radiometrics are not water-masked")
        return None
    import geopandas as gpd
    g = gpd.read_file(hits[0]).to_crs(grid.crs)
    cols = [c for c in ("descriptio", "legend_sho") if c in g.columns]
    if not cols:
        print("  geomorphology layer has no description field; radiometrics are not water-masked")
        return None
    label = g[cols].astype(str).agg(" | ".join, axis=1)
    is_water = label.str.lower().str.contains("|".join(WATER_KEYWORDS))
    g["class"], g["is_water"], g["km2"] = label, is_water, g.area / 1e6
    table = g.groupby(["class", "is_water"]).km2.agg(["sum", "size"]).reset_index()
    table.columns = ["class", "treated_as_water", "area_km2", "polygons"]
    table.sort_values(["treated_as_water", "area_km2"], ascending=[False, False]).to_csv(
        out_dir / "water_classes.csv", index=False)
    m = np.zeros((grid.height, grid.width), dtype=bool)
    if is_water.any():
        m = rasterize(((geom, 1) for geom in g.loc[is_water].geometry), out_shape=m.shape,
                      transform=grid.transform, fill=0, all_touched=False, dtype="uint8").astype(bool)
    used = table[table.treated_as_water]
    print(f"  water mask: {int(m.sum()) * (grid.cell / 1000) ** 2:,.0f} km2 from {len(used)} classes: "
          + "; ".join(used["class"].head(6)) + (" ..." if len(used) > 6 else ""))
    return m


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the V1 raster feature stack.")
    ap.add_argument("raw", help="data/raw/aikosh-gsi-ka-ap")
    ap.add_argument("processed", help="data/processed (holds grid_spec.json and ka_ap_v1.gpkg)")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg", help="GeoPackage name inside processed/")
    args = ap.parse_args()
    raw, proc = Path(args.raw), Path(args.processed)
    grid = Grid(json.loads((proc / "grid_spec.json").read_text()))
    gpkg = proc / args.gpkg
    out = proc / "stack"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    aoi = pyogrio.read_dataframe(gpkg, layer="aoi")
    inside = geometry_mask(aoi.geometry, out_shape=(grid.height, grid.width),
                           transform=grid.transform, invert=True)
    water = water_mask(raw, grid, proc)
    if water is not None:
        prof = grid.profile()
        prof.update(dtype="uint8", nodata=255)
        prof.pop("predictor", None)   # floating-point predictor is invalid for an integer mask
        with rasterio.open(proc / "water_mask.tif", "w", **prof) as dst:
            dst.write(water.astype("uint8"), 1)

    base, meta = {}, {}
    for key, (pattern, units) in SOURCES.items():
        src = find_one(raw, pattern)
        base[key], method = to_grid(src, grid)
        meta[key] = (src.relative_to(raw).as_posix(), method, units)
        print(f"  resampled {key:5} <- {meta[key][0]} ({method})  [{time.time() - t0:.0f}s]", flush=True)

    feats: dict[str, tuple[np.ndarray, str, str, str, str]] = {}  # name -> (array, group, source, method, units)

    def add(name, arr, group, src_key, method, units):
        a = np.asarray(arr, dtype="float32").copy()
        a[~inside | np.isnan(base[src_key])] = np.nan
        feats[name] = (a, group, meta[src_key][0], method, units)

    add("mag_tmi", base["tmi"], "magnetics", "tmi", meta["tmi"][1], "nT")
    for name, arr in magnetic_features(base["tmi"], grid.cell).items():
        add(name, arr, "magnetics", "tmi", "derived from mag_tmi", "degrees" if name == "mag_tilt" else "nT/m")
    print(f"  magnetic derivatives done  [{time.time() - t0:.0f}s]", flush=True)

    add("grav_ba", base["ba"], "gravity", "ba", meta["ba"][1], "mGal")
    g = gravity_features(base["ba"], grid.cell)
    add("grav_residual", g["grav_residual"], "gravity", "ba", "BA minus 10 km Gaussian regional", "mGal")
    add("grav_hgm", g["grav_hgm"], "gravity", "ba", "horizontal gradient of BA", "mGal/km")

    if water is not None:  # radiometrics over water read near zero: no information about the rock
        for key in ("k", "eth", "eu", "dose"):
            base[key][water] = np.nan
    for key, name in [("k", "rad_k"), ("eth", "rad_eth"), ("eu", "rad_eu"), ("dose", "rad_dose")]:
        add(name, base[key], "radiometrics", key, meta[key][1], meta[key][2])
    r = radiometric_features(base["k"], base["eth"], base["eu"])
    units = {"rad_k_eth": "%/ppm", "rad_eu_eth": "ratio", "rad_f": "%", "rad_heat": "microW/m3"}
    for name, arr in r.items():
        arr = np.where(np.isnan(base["k"]) | np.isnan(base["eth"]) | np.isnan(base["eu"]), np.nan, arr)
        how = f"derived from K, eTh, eU; missing where eTh < {RATIO_MIN_ETH:g} ppm" if name != "rad_heat" else "derived from K, eTh, eU"
        add(name, arr, "radiometrics", "eth", how, units[name])

    # ---- gold occurrences vs background (a first look, not a result)
    lab = pyogrio.read_dataframe(gpkg, layer="gold_labels")
    lab = lab[lab["label_class"] == "occurrence"]
    rows, cols = rowcol(grid.transform, lab.geometry.x.values, lab.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    rows, cols = rows[ok], cols[ok]

    n_inside = int(inside.sum())
    manifest = []
    for name, (a, group, src, method, unit) in feats.items():
        write(out / f"{name}.tif", a, grid)
        v = a[inside & np.isfinite(a)]
        at_gold = a[rows, cols]
        at_gold = at_gold[np.isfinite(at_gold)]
        q = np.percentile(v, [0, 1, 50, 99, 100]) if v.size else [np.nan] * 5
        manifest.append({
            "feature": name, "group": group, "source": src, "method": method, "units": unit,
            "coverage_pct": round(100 * v.size / n_inside, 1),
            "min": q[0], "p1": q[1], "p50": q[2], "p99": q[3], "max": q[4],
            "median_at_gold": float(np.median(at_gold)) if at_gold.size else np.nan,
            "gold_percentile": round(100 * (v < np.median(at_gold)).mean(), 0) if at_gold.size and v.size else np.nan,
        })
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        for m in manifest:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in m.items()})

    # ---- quick-look panel
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(feats)
        ncol = 4
        fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(12, 3.4 * np.ceil(n / ncol)), dpi=110)
        for ax, (name, (a, *_)) in zip(axes.flat, feats.items()):
            s = a[::4, ::4]
            lo, hi = np.nanpercentile(s, [2, 98]) if np.isfinite(s).any() else (0, 1)
            ax.imshow(s, vmin=lo, vmax=hi, cmap="viridis", interpolation="nearest")
            ax.scatter(cols / 4, rows / 4, s=4, c="red", linewidths=0)
            ax.set_title(name, fontsize=8)
            ax.set_axis_off()
        for ax in list(axes.flat)[n:]:
            ax.set_axis_off()
        fig.suptitle("V1 raster stack (2-98% stretch; red = gold occurrences)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "stack_quicklook.png")
        quick = "stack_quicklook.png"
    except ImportError:
        quick = "(skipped: matplotlib not installed)"

    # ---- summary
    print(f"\nWrote {len(feats)} layers to {out}  [{time.time() - t0:.0f}s]")
    print(f"{'feature':15} {'cover%':>7} {'p1':>10} {'p50':>10} {'p99':>10} {'gold_pctl':>9}")
    for m in manifest:
        print(f"{m['feature']:15} {m['coverage_pct']:>7} {m['p1']:>10.4g} {m['p50']:>10.4g} "
              f"{m['p99']:>10.4g} {m['gold_percentile']:>9}")
    print("\ngold_pctl = where the median gold occurrence sits in that layer's AOI distribution "
          "(50 = no difference). A first look only; it is not a model result.")
    print(f"Manifest: {out / 'manifest.csv'}   Quick-look: {quick}")


if __name__ == "__main__":
    main()
