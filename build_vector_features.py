#!/usr/bin/env python3
"""
build_vector_features.py - lithology and structure features on the 100 m grid, plus the model mask.

Run from the repository folder, after build_raster_stack.py:
    python build_vector_features.py data/raw/aikosh-gsi-ka-ap data/processed

Writes into data/processed/stack/ (float32, NaN outside the AOI, same grid as the raster stack):
  geo_litho_class           lithology class code (categorical; codes in litho_classes.csv)
  geo_in_greenstone         1 inside a greenstone (supracrustal) belt, 0 outside
  geo_greenstone_sdist      signed distance to the belt margin, m (negative inside)
  geo_dist_contact          distance to any lithology-class contact, m
  geo_dist_bif              distance to iron formation / chert, m
  geo_dist_ultramafic       distance to ultramafic rock, m
  geo_dist_quartz_vein      distance to mapped quartz vein / reef, m
  geo_dist_schist           distance to chlorite-sericite schist (altered shear-zone rock), m
  struct_dist_fault         distance to fault, m
  struct_dist_shear         distance to shear zone, m
  struct_dist_fold          distance to fold axial trace, m
  struct_dist_dyke          distance to dyke (lines + dolerite polygons), m
  struct_dist_fracture      distance to joint / fracture lineament, m
  struct_dens_faultshear    fault + shear zone length within 2 km, km/km2
  struct_dens_lineament     all lineament length within 2 km, km/km2
and:
  data/processed/mask_model.tif               1 = AOI cell with magnetics, radiometrics, gravity and mapped lithology
  data/processed/stack/litho_classes.csv       class codes, areas, gold counts
  data/processed/stack/manifest_vectors.csv    coverage, value ranges, gold-vs-background
  data/processed/stack/vector_quicklook.png    (if matplotlib is installed)

The class mapping lives in litho_classes.py. Only features inside the lon/lat box were supplied,
so distances near the box edge can be overestimated.
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
    from rasterio.features import geometry_mask, rasterize
    from rasterio.transform import Affine, rowcol
    from scipy import ndimage
    from scipy.signal import fftconvolve
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litho_classes as lc  # noqa: E402

DENSITY_RADIUS_M = 2000.0
MASK_LAYERS = ["mag_tmi", "rad_k", "grav_ba"]  # cells must have all three to enter the model


# ----------------------------------------------------------------------------- grid + io

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


def read_vec(raw: Path, pattern: str, crs: str, required: bool = True) -> gpd.GeoDataFrame:
    hits = sorted(raw.glob(pattern))
    if not hits:
        if required:
            sys.exit(f"Not found under {raw}: {pattern}")
        print(f"  (optional layer missing: {pattern})")
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    g = gpd.read_file(hits[0]).to_crs(crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    return g


def polygonal(geom):
    """Keep only polygon parts (make_valid can return mixed collections)."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        return MultiPolygon([p for g in polys for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]) if polys else None
    return None


def presence(geoms, grid: Grid, all_touched: bool = True) -> np.ndarray:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize(((g, 1) for g in geoms), out_shape=grid.shape, transform=grid.transform,
                     fill=0, all_touched=all_touched, dtype="uint8").astype(bool)


def distance(p: np.ndarray, cell: float) -> np.ndarray:
    if not p.any():
        return np.full(p.shape, np.nan, dtype="float32")
    return (ndimage.distance_transform_edt(~p) * cell).astype("float32")


def density(p: np.ndarray, cell: float, radius: float) -> np.ndarray:
    """Approximate line length per area: rasterised line cells x cell length / disk area (km/km2)."""
    r = int(round(radius / cell))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2 <= r * r).astype("float32")
    hits = np.rint(fftconvolve(p.astype("float32"), disk, mode="same"))  # integer counts; drops FFT round-off
    return (np.clip(hits, 0, None) * (cell / 1000.0) / (disk.sum() * (cell / 1000.0) ** 2)).astype("float32")


def lines_where(g: gpd.GeoDataFrame, col: str, values: set) -> list:
    if g.empty or col not in g.columns:
        return []
    return list(g.loc[g[col].astype(str).str.strip().isin(values)].geometry)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Build lithology and structure features.")
    ap.add_argument("raw", help="data/raw/aikosh-gsi-ka-ap")
    ap.add_argument("processed", help="data/processed (holds grid_spec.json, ka_ap_v1.gpkg, stack/)")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    raw, proc = Path(args.raw), Path(args.processed)
    grid = Grid(json.loads((proc / "grid_spec.json").read_text()))
    stack = proc / "stack"
    stack.mkdir(parents=True, exist_ok=True)
    gpkg = proc / args.gpkg
    t0 = time.time()
    U, cell = grid.crs, grid.cell

    aoi = gpd.read_file(gpkg, layer="aoi")
    inside = geometry_mask(aoi.geometry, out_shape=grid.shape, transform=grid.transform, invert=True)

    # ---------------- lithology classes
    lit = read_vec(raw, "50K/lithology*/*.shp", U)
    lit["geometry"] = lit.geometry.make_valid().apply(polygonal)
    lit = lit[lit.geometry.notna()]
    cls = lit["lithologic"].apply(lc.classify)
    lit["code"], lit["cls"] = [c for c, _ in cls], [n for _, n in cls]
    lit["green"] = [lc.is_greenstone(a, b) for a, b in zip(lit.get("supergroup"), lit.get("group_name"))]
    lit["area"] = lit.area

    # class = rock at the cell centre. Thin units (dykes, veins, iron formation) are not forced into the
    # class map, where they would overwrite host rock; they enter the model through distance layers.
    codes = np.zeros(grid.shape, dtype="uint8")
    ordered = lit.sort_values("area", ascending=False)
    rasterize(((g, c) for g, c in zip(ordered.geometry, ordered.code)), out=codes,
              transform=grid.transform, all_touched=False)
    mapped = codes > 0
    print(f"  lithology rasterised: {mapped[inside].mean():.1%} of AOI mapped  [{time.time() - t0:.0f}s]", flush=True)

    green = presence(lit.loc[lit.green, "geometry"], grid, all_touched=False)
    sdist = np.where(green, -distance(~green, cell), distance(green, cell))

    edge = np.zeros(grid.shape, dtype=bool)
    for shift in ((0, 1), (1, 0)):
        a = codes[: grid.height - shift[0], : grid.width - shift[1]]
        b = codes[shift[0]:, shift[1]:]
        diff = (a != b) & (a > 0) & (b > 0)
        edge[: grid.height - shift[0], : grid.width - shift[1]] |= diff
        edge[shift[0]:, shift[1]:] |= diff

    def cls_presence(names):
        return presence(lit.loc[lit.cls.isin(names), "geometry"], grid, all_touched=True)

    # ---------------- structure
    f50 = read_vec(raw, "50K/fault*/*.shp", U)
    f25 = read_vec(raw, "25K/fault*/*.shp", U, required=False)
    sh25 = read_vec(raw, "25K/shear*/*.shp", U, required=False)
    fo50 = read_vec(raw, "50K/fold*/*.shp", U, required=False)
    fo25 = read_vec(raw, "25K/fold*/*.shp", U, required=False)
    dk25 = read_vec(raw, "25K/dyke*/*.shp", U, required=False)
    lform = read_vec(raw, "Lineament_250k/lineament_form.shp", U, required=False)
    l250 = read_vec(raw, "Lineament_250k/lineament_250k_ngdr.shp", U, required=False)

    def lin(values):
        return lines_where(lform, "LINEAMENT_", values) + lines_where(l250, "l2descript", values)

    faults = list(f50.geometry) + list(f25.geometry) + lin({"Fault"})
    shears = list(sh25.geometry) + lin({"Shear zone"})
    folds = list(fo50.geometry) + list(fo25.geometry) + lin({"Axial trace of fold"})
    dykes = list(dk25.geometry) + lin({"Dyke"}) + list(lit.loc[lit["lithologic"].astype(str).str.upper() == "DOLERITE", "geometry"])
    fractures = lin({"Joint/Fracture"})

    feats = {}  # name -> (array, group, kind, source, units)
    feats["geo_litho_class"] = (np.where(mapped, codes, np.nan), "lithology", "categorical", "50K lithology", "code")
    feats["geo_in_greenstone"] = (np.where(mapped, green.astype("float32"), np.nan), "lithology", "binary", "50K supergroup/group", "0/1")
    feats["geo_greenstone_sdist"] = (sdist, "lithology", "continuous", "50K supergroup/group", "m")
    feats["geo_dist_contact"] = (distance(edge, cell), "lithology", "continuous", "50K lithology classes", "m")
    feats["geo_dist_bif"] = (distance(cls_presence(["bif_chert"]), cell), "lithology", "continuous", "50K lithology", "m")
    feats["geo_dist_ultramafic"] = (distance(cls_presence(["ultramafic"]), cell), "lithology", "continuous", "50K lithology", "m")
    feats["geo_dist_quartz_vein"] = (distance(cls_presence(["quartz_vein"]), cell), "lithology", "continuous", "50K lithology", "m")
    feats["geo_dist_schist"] = (distance(cls_presence(["chl_ser_schist"]), cell), "lithology", "continuous", "50K lithology", "m")
    print(f"  lithology features done  [{time.time() - t0:.0f}s]", flush=True)

    feats["struct_dist_fault"] = (distance(presence(faults, grid), cell), "structure", "continuous", "50K+25K faults, lineaments", "m")
    feats["struct_dist_shear"] = (distance(presence(shears, grid), cell), "structure", "continuous", "25K shear zones, lineaments", "m")
    feats["struct_dist_fold"] = (distance(presence(folds, grid), cell), "structure", "continuous", "50K+25K folds, lineaments", "m")
    feats["struct_dist_dyke"] = (distance(presence(dykes, grid), cell), "structure", "continuous", "dykes + dolerite polygons", "m")
    feats["struct_dist_fracture"] = (distance(presence(fractures, grid), cell), "structure", "continuous", "joint/fracture lineaments", "m")
    feats["struct_dens_faultshear"] = (density(presence(faults + shears, grid, all_touched=False), cell, DENSITY_RADIUS_M),
                                       "structure", "continuous", "faults + shear zones", "km/km2")
    feats["struct_dens_lineament"] = (density(presence(list(lform.geometry), grid, all_touched=False), cell, DENSITY_RADIUS_M),
                                      "structure", "continuous", "all lineaments (form set)", "km/km2")
    print(f"  structure features done  [{time.time() - t0:.0f}s]", flush=True)

    # ---------------- model mask
    mask = inside & mapped
    used = []
    for name in MASK_LAYERS:
        p = stack / f"{name}.tif"
        if p.exists():
            with rasterio.open(p) as src:
                mask &= np.isfinite(src.read(1))
            used.append(name)
        else:
            print(f"  warning: {p.name} not found; mask ignores it")
    write(proc / "mask_model.tif", mask.astype("uint8"), grid, dtype="uint8", nodata=255)

    # ---------------- gold coverage + write features
    lab = gpd.read_file(gpkg, layer="gold_labels")
    rr, cc = rowcol(grid.transform, lab.geometry.x.values, lab.geometry.y.values)
    rr, cc = np.asarray(rr), np.asarray(cc)
    ok = (rr >= 0) & (rr < grid.height) & (cc >= 0) & (cc < grid.width)
    lab["in_mask"] = False
    lab.loc[ok, "in_mask"] = mask[rr[ok], cc[ok]]
    occ = lab["label_class"].eq("occurrence").values & ok
    orr, occ_c = rr[occ], cc[occ]

    manifest = []
    for name, (a, group, kind, src, unit) in feats.items():
        a = np.asarray(a, dtype="float32").copy()
        a[~inside] = np.nan
        write(stack / f"{name}.tif", a, grid)
        v = a[mask & np.isfinite(a)]
        g = a[orr, occ_c]
        g = g[np.isfinite(g)]
        q = np.percentile(v, [1, 50, 99]) if v.size else [np.nan] * 3
        manifest.append({
            "feature": name, "group": group, "type": kind, "source": src, "units": unit,
            "coverage_pct": round(100 * np.isfinite(a[inside]).mean(), 1),
            "p1": q[0], "p50": q[1], "p99": q[2],
            "median_at_gold": float(np.median(g)) if g.size else np.nan,
            # mid-rank percentile, so ties (e.g. many zero-density cells) don't read as 0 or 100
            "gold_percentile": round(100 * ((v < np.median(g)).mean() + 0.5 * (v == np.median(g)).mean()), 0)
            if (g.size and v.size and kind == "continuous") else "",
        })
    with open(stack / "manifest_vectors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        for m in manifest:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in m.items()})

    # class table
    occ_codes = codes[orr, occ_c]
    with open(stack / "litho_classes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "class", "area_km2_in_mask", "area_share_pct", "gold_occurrences", "rock_types"])
        tot = mask.sum()
        for code, name, rocks in lc.CLASSES + [(lc.OTHER[0], lc.OTHER[1], ["(unlisted)"])]:
            n = int(((codes == code) & mask).sum())
            w.writerow([code, name, round(n * (cell / 1000) ** 2, 1), round(100 * n / tot, 2),
                        int((occ_codes == code).sum()), "; ".join(rocks)])

    # quick-look
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(feats)
        fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(12, 3.4 * np.ceil(n / 4)), dpi=110)
        for ax, name in zip(axes.flat, feats):
            with rasterio.open(stack / f"{name}.tif") as src:
                s = src.read(1)[::4, ::4]
            s = np.where(mask[::4, ::4], s, np.nan)
            if name == "geo_litho_class":
                ax.imshow(s, cmap="tab20", vmin=0.5, vmax=12.5, interpolation="nearest")
            else:
                lo, hi = np.nanpercentile(s, [2, 98]) if np.isfinite(s).any() else (0, 1)
                ax.imshow(s, vmin=lo, vmax=hi, cmap="viridis_r" if "dist" in name else "viridis", interpolation="nearest")
            ax.scatter(occ_c / 4, orr / 4, s=4, c="red", linewidths=0)
            ax.set_title(name, fontsize=8)
            ax.set_axis_off()
        for ax in list(axes.flat)[n:]:
            ax.set_axis_off()
        fig.suptitle("Lithology and structure features (inside model mask; red = gold occurrences)", fontsize=10)
        fig.tight_layout()
        fig.savefig(stack / "vector_quicklook.png")
        quick = "vector_quicklook.png"
    except ImportError:
        quick = "(skipped: matplotlib not installed)"

    # summary
    print(f"\nWrote {len(feats)} layers to {stack} and {proc / 'mask_model.tif'}  [{time.time() - t0:.0f}s]")
    print(f"Model mask: {mask.sum() * (cell / 1000) ** 2:,.0f} km2 = {mask[inside].mean():.1%} of AOI "
          f"(needs mapped lithology + {', '.join(used) or 'no stack layers'})")
    print(f"Gold labels inside mask: {int(lab.in_mask.sum())} of {len(lab)} "
          f"(occurrences {int(lab.loc[lab.label_class == 'occurrence', 'in_mask'].sum())} of "
          f"{int((lab.label_class == 'occurrence').sum())})")
    out_ids = lab.loc[~lab.in_mask, "label_id"].tolist()
    if out_ids:
        print(f"  outside mask: {', '.join(out_ids)}")
    print(f"\n{'feature':24} {'cover%':>7} {'p50':>10} {'gold_med':>10} {'gold_pctl':>9}")
    for m in manifest:
        print(f"{m['feature']:24} {m['coverage_pct']:>7} {m['p50']:>10.4g} {m['median_at_gold']:>10.4g} {str(m['gold_percentile']):>9}")
    print(f"\nClass table: {stack / 'litho_classes.csv'}   Quick-look: {quick}")


if __name__ == "__main__":
    main()
