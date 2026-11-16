#!/usr/bin/env python3
"""
prepare_vectors.py - build the V1 gold label set and clean vector inputs for the KA/AP block.

Run from the repository folder:
    python prepare_vectors.py data/raw/aikosh-gsi-ka-ap data/processed/ka_ap_v1.gpkg

Install (same env as inventory.py, plus geopandas, scipy and openpyxl):
    conda install -n te -c conda-forge geopandas scipy openpyxl matplotlib

Writes one GeoPackage (EPSG:32643, UTM 43N) with these layers:
    aoi                  study-area box (76.0-77.75 E, 13.75-15.75 N), snapped to the 100 m grid
    gold_labels          positive labels: mapped gold occurrences + merged old workings, with weight and cluster_id
    occurrences_all      every mapped mineral occurrence (all commodities), 25K duplicates removed
    exploration_blocks   survey areas: sampling-bias layer and index into the GSI technical reports
    geochem_ngcm         stream-sediment geochemistry with corrected units and unanalysed values set to NaN
and grid_spec.json next to the GeoPackage.

Decisions encoded here (see notes printed at the end):
  * 25K mineralization points duplicate 50K points exactly, so 50K is the occurrence source.
  * 50K "old workings" have no commodity field; every one lies within 5 km of a mapped gold
    occurrence (median ~300 m) and nothing else is mapped in their belt, so they are treated as
    gold, flagged "inferred", merged within 100 m, and down-weighted (one belt, many pits).
  * Exploration blocks are survey areas (median ~150 km2, mostly G4 reconnaissance), not deposits,
    so they are never positive labels.
  * Geochemistry: the .xlsx and the shapefile hold identical values; the .xlsx unit labels are the
    plausible ones (REE in ppm, Ag/Cd in ppb). Elements that are 0 for an entire toposheet were
    never analysed there and become NaN. Gold and its pathfinders are missing on 19 western sheets.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import geopandas as gpd
    from scipy.spatial import cKDTree
    from shapely.geometry import box
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. See the install line at the top of this file.")

UTM = "EPSG:32643"
AOI_LONLAT = (76.0, 13.75, 77.75, 15.75)
CELL = 100.0                    # modelling grid cell, metres
MERGE_OLD_WORKINGS_M = 100.0    # merge old-working pits closer than one cell
CLUSTER_LINK_M = 5000.0         # labels within 5 km share a cluster (use as spatial-CV groups)
OLD_WORKING_WEIGHT = 0.25       # tunable: 127 pits in one belt vs 48 occurrences across the area
UNANALYSED_SHARE = 0.95         # element counts as unanalysed on a toposheet if >=95% of values are 0


# ----------------------------------------------------------------------------- helpers

def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if not hits:
        sys.exit(f"Not found under {root}: {pattern}")
    return hits[0]


def union_find_groups(xy: np.ndarray, radius: float) -> np.ndarray:
    """Connected components of points linked when closer than `radius`."""
    parent = np.arange(len(xy))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(xy).query_pairs(radius):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([root(i) for i in range(len(xy))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def xy_of(g: gpd.GeoDataFrame) -> np.ndarray:
    return np.c_[g.geometry.x.values, g.geometry.y.values]


# ----------------------------------------------------------------------------- layers

def build_aoi() -> tuple[gpd.GeoDataFrame, dict]:
    poly = gpd.GeoSeries([box(*AOI_LONLAT)], crs="EPSG:4326").to_crs(UTM).iloc[0]
    x0, y0, x1, y1 = poly.bounds
    x0, y0 = np.floor(x0 / CELL) * CELL, np.floor(y0 / CELL) * CELL
    x1, y1 = np.ceil(x1 / CELL) * CELL, np.ceil(y1 / CELL) * CELL
    spec = {
        "crs": UTM, "cell_m": CELL,
        "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1,
        "width": int(round((x1 - x0) / CELL)), "height": int(round((y1 - y0) / CELL)),
        "transform_gdal": [x0, CELL, 0.0, y1, 0.0, -CELL],
        "aoi_lonlat": list(AOI_LONLAT),
        "note": "Rasterise every layer onto this grid. Cells outside the lon/lat AOI polygon are masked.",
    }
    aoi = gpd.GeoDataFrame({"name": ["KA-AP GSI block"], "area_km2": [poly.area / 1e6]},
                           geometry=[poly], crs=UTM)
    return aoi, spec


def build_occurrences(raw: Path) -> gpd.GeoDataFrame:
    m50 = gpd.read_file(find_one(raw, "50K/mineralization*/*.shp")).to_crs(UTM)
    m25 = gpd.read_file(find_one(raw, "25K/mineralization*/*.shp")).to_crs(UTM)
    m50 = m50.assign(commodity=m50["commodity"].str.strip().str.title(), source="50K mineralization")
    m25 = m25.assign(commodity=m25["commodity"].str.strip().str.title(), source="25K mineralization")
    # drop 25K points that duplicate a 50K point of the same commodity (within 50 m)
    keep = []
    for i, r in m25.iterrows():
        same = m50[m50.commodity == r.commodity]
        keep.append(same.empty or same.distance(r.geometry).min() > 50)
    m25_new = m25[np.array(keep, dtype=bool)]
    cols = ["commodity", "source", "toposheet_", "geometry"]
    occ = pd.concat([m50[cols], m25_new[cols]], ignore_index=True)
    occ = gpd.GeoDataFrame(occ.rename(columns={"toposheet_": "toposheet"}), crs=UTM)
    occ.attrs["dropped_25k_duplicates"] = int((~np.array(keep, dtype=bool)).sum())
    return occ


def build_gold_labels(raw: Path, occ: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gold = occ[occ.commodity == "Gold"].copy()
    gold = gold.assign(label_class="occurrence", confidence="mapped", n_merged=1, weight=1.0)

    ow = gpd.read_file(find_one(raw, "50K/mine_quarry*/*.shp")).to_crs(UTM)
    d_gold = ow.geometry.apply(lambda g: gold.distance(g).min())
    ow = ow.assign(dist_to_gold_m=d_gold)
    # merge pits closer than one cell into a single label at their centroid
    grp = union_find_groups(xy_of(ow), MERGE_OLD_WORKINGS_M)
    merged = ow.assign(_g=grp).dissolve(by="_g", aggfunc={"dist_to_gold_m": "min", "status": "first"})
    merged["n_merged"] = ow.assign(_g=grp).groupby("_g").size()
    merged["geometry"] = merged.geometry.centroid
    merged = merged.reset_index(drop=True).assign(
        commodity="Gold", source="50K mine/quarry (old workings)", toposheet=None,
        label_class="old_working", confidence="inferred: nearest mapped occurrence is gold",
        weight=OLD_WORKING_WEIGHT)

    cols = ["commodity", "label_class", "confidence", "source", "toposheet", "n_merged", "weight", "geometry"]
    lab = gpd.GeoDataFrame(pd.concat([gold[cols], merged[cols]], ignore_index=True), crs=UTM)
    lab["cluster_id"] = union_find_groups(xy_of(lab), CLUSTER_LINK_M)
    lab["label_id"] = [f"AU{i:03d}" for i in range(1, len(lab) + 1)]
    lab.attrs["old_workings_raw"] = len(ow)
    lab.attrs["old_workings_max_dist_to_gold_m"] = float(d_gold.max())
    return lab[["label_id"] + cols[:-1] + ["cluster_id", "geometry"]]


def build_blocks(raw: Path) -> gpd.GeoDataFrame:
    ex = gpd.read_file(find_one(raw, "Exploration_data/*.shp")).to_crs(UTM)
    ex["geometry"] = ex.geometry.make_valid()
    out = gpd.GeoDataFrame({
        "report_id": ex["exp_upid"],
        "title": ex["project_ti"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "commodity": ex["commodity"],
        "has_gold": ex["commodity"].str.contains("gold", case=False, na=False),
        "stage": ex["explorat_2"],
        "agency": ex["name_of_ex"],
        "year_start": pd.to_numeric(ex["period_of_"], errors="coerce"),
        "year_end": pd.to_numeric(ex["period_o_1"], errors="coerce"),
        "map_scale": ex["scale"],
        "districts": ex["district_n"],
        "report_files": ex["exp_rep_do"],
        "gis_files": ex["explorat_3"],
        "table_files": ex["explorat_4"],
        "area_km2": ex.area / 1e6,
    }, geometry=ex.geometry, crs=UTM)
    # contact numbers, uploader ids and image paths are deliberately not carried over
    return out


def build_geochem(raw: Path) -> gpd.GeoDataFrame:
    xl = pd.read_excel(find_one(raw, "Geochemical data_GIS/*.xlsx"))
    xl = xl.drop(columns=[c for c in ("FID", "Shape *", "gid", "objectid") if c in xl.columns])
    xl = xl.rename(columns={"Si02_%": "SiO2_%"})
    # two different samples share one id: keep both, make ids unique
    dup = xl["sampleno"].duplicated(keep="first")
    xl.loc[dup, "sampleno"] = xl.loc[dup, "sampleno"].astype(str) + "_b"

    value_cols = [c for c in xl.columns if c.endswith(("_%", "_ppm", "_ppb")) or c == "LOI_%"]
    xl[value_cols] = xl[value_cols].apply(pd.to_numeric, errors="coerce")
    unanalysed = {}
    for c in value_cols:
        share_zero = (xl[c] == 0).groupby(xl["toposheet"]).mean()
        sheets = share_zero[share_zero >= UNANALYSED_SHARE].index
        if len(sheets):
            mask = xl["toposheet"].isin(sheets)
            xl.loc[mask, c] = np.nan
            unanalysed[c] = len(sheets)
    xl["trace_analysed"] = xl["Au_ppb"].notna()

    g = gpd.GeoDataFrame(xl, geometry=gpd.points_from_xy(xl["X"], xl["Y"]), crs="EPSG:4326").to_crs(UTM)
    g = g.rename(columns={"X": "lon", "Y": "lat"})
    g.attrs["unanalysed"] = unanalysed
    return g


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Build V1 gold labels and clean vector inputs.")
    ap.add_argument("raw", help="data/raw/aikosh-gsi-ka-ap")
    ap.add_argument("out", help="output GeoPackage, e.g. data/processed/ka_ap_v1.gpkg")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    aoi, spec = build_aoi()
    occ = build_occurrences(raw)
    lab = build_gold_labels(raw, occ)
    blocks = build_blocks(raw)
    geo = build_geochem(raw)

    for name, layer in [("aoi", aoi), ("gold_labels", lab), ("occurrences_all", occ),
                        ("exploration_blocks", blocks), ("geochem_ngcm", geo)]:
        layer.to_file(out, layer=name, driver="GPKG")
    (out.parent / "grid_spec.json").write_text(json.dumps(spec, indent=2))

    # ---- summary
    gold_trace = gpd.sjoin_nearest(lab[["label_class", "geometry"]],
                                   geo[["trace_analysed", "geometry"]], how="left")
    gold_trace = gold_trace[~gold_trace.index.duplicated()]
    print(f"Wrote {out} and {out.parent / 'grid_spec.json'}\n")
    print(f"Grid: {spec['width']} x {spec['height']} cells at {CELL:.0f} m "
          f"({spec['width'] * spec['height'] / 1e6:.1f}M); AOI {aoi.area_km2.iloc[0]:,.0f} km2")
    print(f"Occurrences: {len(occ)} (dropped {occ.attrs['dropped_25k_duplicates']} 25K duplicates)")
    print("  by commodity:", occ.commodity.value_counts().head(8).to_dict())
    print(f"Gold labels: {len(lab)}  = {(lab.label_class == 'occurrence').sum()} mapped occurrences "
          f"+ {(lab.label_class == 'old_working').sum()} merged old workings "
          f"(from {lab.attrs['old_workings_raw']} pits; max distance to a gold occurrence "
          f"{lab.attrs['old_workings_max_dist_to_gold_m']:.0f} m)")
    print(f"  effective weight: {lab.weight.sum():.1f} | spatial clusters (5 km link): {lab.cluster_id.nunique()}")
    sizes = lab.groupby("cluster_id").size().sort_values(ascending=False)
    print(f"  largest clusters: {sizes.head(5).to_dict()}")
    print(f"  occurrences in sheets without gold/pathfinder geochem: "
          f"{((gold_trace.label_class == 'occurrence') & ~gold_trace.trace_analysed).sum()}")
    print(f"Exploration blocks: {len(blocks)} (gold-related {int(blocks.has_gold.sum())}); "
          f"stages {blocks.stage.value_counts().to_dict()}")
    print(f"Geochem: {len(geo)} samples; trace elements analysed on {int(geo.trace_analysed.sum())}")
    print(f"  set to NaN on unanalysed sheets: {geo.attrs['unanalysed']}")


if __name__ == "__main__":
    main()
