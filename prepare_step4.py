#!/usr/bin/env python3
"""
prepare_step4.py - build the desk-check sheet and the GSI report pull list for the tier A/B prospects.

Run from the repository folder, after run_v1.bat:
    python prepare_step4.py data/processed data/raw/aikosh-gsi-ka-ap --model pg_ens
You normally do not need to run this yourself: auto_desk_check.py runs it first, then fills in
check_status automatically from satellite and map data.

Reads   models/<model>/targets.gpkg, shortlist.csv, prospects.csv, reports_to_pull.csv,
        data/processed/ka_ap_v1.gpkg (exploration_blocks), raw Geomorphology_250k (optional)
Writes  models/<model>/step4/
          desk_check_sheet.csv      one row per target to check: what to look for, a satellite link,
                                    and blank check_status / notes columns to fill in
          desk_check.gpkg           same targets as polygons + peak points (open in QGIS)
          desk_check_targets.kml    polygons for Google Earth Pro
          desk_check_peaks.kml      peak points for Google Earth Pro
          report_pull_list.csv      GSI gold reports for tier A prospects, in download batches,
                                    with the exact file names to fetch
          STEP4_README.txt          check codes and instructions
        data/processed/report_findings.csv   one row per GSI report to read, to fill in as you read them.
                                    Shared by all model runs; new reports are appended, your entries kept.

check_status codes (type one per row):
  keep          rock or soil ground, nothing suspicious
  outcrop       good exposed rock (ridges, rocky knolls) - keep, easy to sample
  old_workings  pits, trenches, shafts or dumps - keep, strong positive
  quarry        stone quarry or crusher - probably an artefact, downgrade
  drop_water    tank, reservoir, river bed or seasonally flooded ground
  drop_urban    village, town, industrial site, highway or rail yard
  drop_alluvium thick soil, alluvium or sand sheet hiding the bedrock
  unclear       cannot tell from imagery

report_findings.csv - outcome codes (these become V2 training labels):
  positive      gold mineralisation confirmed in rock, trench or drill core (working cut-off: any assay
                >= 0.5 g/t Au, or the report describes a gold-bearing zone). Becomes a gold label.
  barren        systematic sampling, trenching or drilling found nothing above ~0.1 g/t Au. Becomes
                "explored, no gold" ground, which V2 uses as real negatives instead of random background.
  inconclusive  reconnaissance only, or results too sparse to call
  not_relevant  the report covers other ground or another commodity
  Fill lon/lat with the location of the best result (positive) or the centre of the tested ground
  (barren), best_result as text ("3.2 g/t over 2 m, trench TR-4"), and page_ref so it can be checked.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
try:
    import geopandas as gpd
    import pandas as pd
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

LOOK_FOR = {
    "low_radiometric": "Low radioelements: tank bed, wet paddy or black soil (drop) vs dark mafic outcrop (keep). "
                       "Use Google Earth history to spot seasonal water.",
    "ratio_extreme": "Unstable ratio: check for water, thin soil or very dark rock.",
    "felsic_radiometric": "High K/Th: pinkish granite boulders or tors mean granite, not greenstone (downgrade).",
    "edge": "On the data edge: note whether the ridge or belt continues outside the block.",
    "geophysics_only": "No gold geochemistry here: note road access and outcrop for future sampling.",
}
DEFAULT_LOOK = "Look for outcrop, old pits or dumps, quarries, villages and water."
BATCH1_MIN_DIST_KM = 5.0   # tier A prospects at least this far from known gold count as new ground


def maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/@?api=1&map_action=map&center={lat:.5f},{lon:.5f}&zoom=16&basemap=satellite"


def basenames(paths) -> str:
    s = str(paths or "")
    if s in ("", "None", "nan"):
        return ""
    return "; ".join(p.strip().split("/")[-1] for p in s.split(",") if p.strip())


def build(proc: Path, raw: str = "", model: str = "pg_ens", gpkg: str = "ka_ap_v1.gpkg") -> Path:
    """Write the step-4 files and return the step4 folder. Also called by auto_desk_check.py."""
    args = argparse.Namespace(processed=str(proc), raw=raw, model=model, gpkg=gpkg)
    proc = Path(args.processed)
    mdir = proc / "models" / args.model
    out = mdir / "step4"
    out.mkdir(parents=True, exist_ok=True)

    tg = gpd.read_file(mdir / "targets.gpkg")
    sl = pd.read_csv(mdir / "shortlist.csv")
    pr = pd.read_csv(mdir / "prospects.csv")
    ab = pr[pr["best_tier"].isin(["A", "B"])][["prospect", "best_tier"]]
    order = {p: i for i, p in enumerate(pr["prospect"])}

    sel = sl.merge(ab, on="prospect")
    sel = sel.merge(tg[["target", "geometry"]], on="target", how="left")
    sel = gpd.GeoDataFrame(sel, geometry="geometry", crs=tg.crs)
    sel["_o"] = sel["prospect"].map(order)
    sel = sel.sort_values(["_o", "mean_pct"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    sel.insert(0, "order", range(1, len(sel) + 1))

    def look(flags):
        fl = [f.strip() for f in str(flags or "").split(",") if f.strip() and f.strip() != "nan"]
        return " | ".join(LOOK_FOR[f] for f in fl if f in LOOK_FOR) or DEFAULT_LOOK
    sel["look_for"] = sel["flags"].apply(look)

    # geomorphology: dominant mapped landform, and whether a quarry / mine dump touches the target
    sel["geomorph"], sel["quarry_or_dump"] = "", False
    gm_hits = sorted(Path(args.raw).glob("Geomorphology*/*.shp")) if args.raw else []
    if gm_hits:
        gm = gpd.read_file(gm_hits[0]).to_crs(tg.crs)
        col = "descriptio" if "descriptio" in gm.columns else gm.columns[0]
        for i, geom in enumerate(sel.geometry):
            if geom is None:
                continue
            hit = gm[gm.intersects(geom)]
            if hit.empty:
                continue
            area = hit.geometry.intersection(geom).area
            sel.at[i, "geomorph"] = str(hit[col].iloc[int(np.argmax(area.values))])
            sel.at[i, "quarry_or_dump"] = bool(hit[col].astype(str).str.contains("quarry|dump", case=False).any())
        sel.loc[sel["quarry_or_dump"], "look_for"] += " | Geomorphology maps a quarry or mine dump here: old workings (keep) or stone quarry (downgrade)?"
    else:
        print("  geomorphology map not given or not found; geomorph columns left blank")

    sel["satellite_link"] = [maps_link(la, lo) for la, lo in zip(sel["peak_lat"], sel["peak_lon"])]
    sel["check_status"], sel["notes"] = "", ""
    cols = ["order", "prospect", "best_tier", "target", "tier", "status", "flags", "look_for", "geomorph",
            "quarry_or_dump", "mean_pct", "stability"] + [c for c in ("pct_min_runs",) if c in sel.columns] + [
            "dist_known_gold_km", "au_geochem_pct", "evidence", "peak_lat", "peak_lon", "satellite_link",
            "check_status", "notes"]
    sel[cols].to_csv(out / "desk_check_sheet.csv", index=False, encoding="utf-8-sig")

    # GIS outputs
    (out / "desk_check.gpkg").unlink(missing_ok=True)
    gcols = [c for c in cols if c != "satellite_link"]
    sel[gcols + ["geometry"]].to_file(out / "desk_check.gpkg", layer="targets_to_check", driver="GPKG")
    peaks = gpd.GeoDataFrame(sel[gcols].copy(), geometry=gpd.points_from_xy(sel["peak_lon"], sel["peak_lat"]),
                             crs="EPSG:4326")
    peaks.to_crs(tg.crs).to_file(out / "desk_check.gpkg", layer="peaks", driver="GPKG")

    def kml_frame(gdf):
        k = gdf.to_crs("EPSG:4326").copy()
        k["Name"] = k["target"] + " (" + k["prospect"] + ", tier " + k["best_tier"] + ")"
        k["Description"] = ("status: " + k["status"].astype(str) + "\nflags: " + k["flags"].fillna("").astype(str)
                            + "\nevidence: " + k["evidence"].astype(str) + "\nlook for: " + k["look_for"])
        return k[["Name", "Description", "geometry"]]
    for name, gdf in (("desk_check_targets.kml", sel), ("desk_check_peaks.kml", peaks)):
        (out / name).unlink(missing_ok=True)
        try:
            kml_frame(gdf).to_file(out / name, driver="KML")
        except Exception as e:  # KML driver missing in some GDAL builds
            print(f"  could not write {name} ({type(e).__name__}); use desk_check.gpkg in QGIS instead")

    # report pull list for tier A prospects
    rp = pd.read_csv(mdir / "reports_to_pull.csv")
    pa = pr[pr["best_tier"] == "A"].set_index("prospect")
    new_a = set(pa.index[pa["dist_known_gold_km"] >= BATCH1_MIN_DIST_KM])
    rows = []
    for _, r in rp.iterrows():
        ps = [p for p in str(r["prospects"]).split() if p in pa.index]
        if not ps:
            continue
        if r["scale"] == "regional":
            batch = 3
        elif any(p in new_a for p in ps):
            batch = 1
        else:
            batch = 2
        rows.append({"batch": batch, "report_id": r["report_id"], "scale": r["scale"], "stage": r["stage"],
                     "year": r["year_start"], "area_km2": r["area_km2"], "tier_a_prospects": " ".join(ps),
                     "title": r["title"], "ngdr_folder": f"Exploration_Data/{r['report_id']}/",
                     "text_files": basenames(r["report_files"]), "table_files": basenames(r["table_files"]),
                     "gis_files": basenames(r["gis_files"]), "downloaded": ""})
    pl = pd.DataFrame(rows)
    if not pl.empty:
        sc = {"detailed": 0, "local": 1, "regional": 2}
        # one report can cover several exploration blocks: keep one row, earliest batch, all prospects
        pl = pl.assign(_s=pl["scale"].map(sc)).sort_values(["batch", "_s", "area_km2"])
        merged_ps = pl.groupby("report_id")["tier_a_prospects"].apply(
            lambda s: " ".join(sorted(set(" ".join(s).split()))))
        pl = pl.drop_duplicates("report_id", keep="first").copy()
        pl["tier_a_prospects"] = pl["report_id"].map(merged_ps)
        pl = pl.drop(columns="_s")
        pl.to_csv(out / "report_pull_list.csv", index=False, encoding="utf-8-sig")

    if not pl.empty:
        update_findings(proc / "report_findings.csv", pl)
    (out / "STEP4_README.txt").write_text(__doc__.strip() + "\n", encoding="utf-8")
    n_a = int((sel["best_tier"] == "A").sum())
    print(f"Desk check: {len(sel)} targets in {sel['prospect'].nunique()} tier A/B prospects "
          f"({n_a} targets in tier A prospects first)")
    print(f"  flagged: {int(sel['flags'].fillna('').str.len().gt(0).sum())}, "
          f"touching a mapped quarry/mine dump: {int(sel['quarry_or_dump'].sum())}")
    if not pl.empty:
        for b, g in pl.groupby("batch"):
            label = {1: "new-ground tier A", 2: "other tier A", 3: "regional (later)"}[b]
            print(f"  report batch {b} ({label}): {len(g)} reports - {', '.join(g['report_id'])}")
    print(f"Outputs in {out}")
    return out


FINDINGS_COLS = ["report_id", "batch", "tier_a_prospects", "year", "scale", "stage", "title", "reviewed",
                 "outcome", "best_result", "sample_type", "lon", "lat", "page_ref", "notes"]


def update_findings(path: Path, pl: pd.DataFrame) -> None:
    """Create or extend the report findings sheet without touching rows already filled in."""
    new = pl.reindex(columns=FINDINGS_COLS)
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        add = new[~new["report_id"].isin(old["report_id"])]
        if add.empty:
            return
        out = pd.concat([old, add.astype(str).replace("nan", "")], ignore_index=True)
        print(f"  report_findings.csv: {len(add)} new reports added, {len(old)} kept")
    else:
        out = new
        print(f"  report_findings.csv created with {len(out)} reports to read")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the step-4 desk check and report pull list.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("raw", nargs="?", default="", help="data/raw/aikosh-gsi-ka-ap (for the geomorphology map)")
    ap.add_argument("--model", default="pg_ens")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    build(Path(args.processed), args.raw, args.model, args.gpkg)


if __name__ == "__main__":
    main()
