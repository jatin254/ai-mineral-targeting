#!/usr/bin/env python3
"""
shortlist_targets.py - group targets into prospects, tier them, and list the GSI reports to pull.

Run from the repository folder, after make_targets.py:
    python shortlist_targets.py data/processed --model pathf_geophys

Reads   data/processed/models/<model>/targets.gpkg, data/processed/ka_ap_v1.gpkg (aoi, exploration_blocks)
Writes  data/processed/models/<model>/
          shortlist.csv          every target with tier and flags
          prospects.gpkg/.geojson  targets within 2 km grouped into prospects (EPSG:32643 / lon-lat)
          prospects.csv          one row per prospect, best first
          reports_to_pull.csv    GSI gold exploration reports covering Tier A/B prospects, with file paths
          prospects.png          map

Tiers (status "known" targets are kept as context and never tiered A/B):
  A  new or near-known, stability >= 0.75, gold geochemistry covers >= 50% and supports it
     (Au, As, Sb, Bi or the index elevated), no artefact flag
  B  new or near-known, stability >= 0.60, no artefact flag (includes geophysics-only targets)
  C  everything else that is not known: unstable, or flagged
Flags (desk-check before any field work):
  ratio_extreme   eU/eTh or K/eTh z >= 5: thorium near zero makes the ratio unstable
  low_radiometric two or more of eU, eTh, K, dose rate, radiogenic heat strongly low: mafic rock,
                  or a water body / tank / thin soil. Check satellite imagery.
  felsic_radiometric two or more of K, eTh, dose rate, radiogenic heat strongly high: likely granite
                  inside the 1 km belt buffer rather than greenstone
  edge            within 2 km of the data block edge: ground beyond it is unmodelled
  geophysics_only no gold geochemistry; confirm with stream-sediment or soil samples
"""
from __future__ import annotations

import argparse
import re
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

GROUP_M = 1000.0   # targets closer than this form one prospect (known-gold targets excluded)
GEO_POS = {"Au", "As", "Sb", "Bi", "Au pathfinder index"}
RAD = {"eU", "eTh", "K", "dose rate", "radiogenic heat"}
STAGE_ORDER = {"E1": 0, "E2": 1, "P1": 2, "P2": 3, "G1": 4, "G2": 5, "G3": 6, "G4": 7}


def parse_evidence(ev: str) -> list[tuple[str, float]]:
    out = []
    for part in str(ev or "").split(";"):
        m = re.match(r"\s*(.+?)\s([+-]\d+(?:\.\d+)?)$", part)
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Shortlist prospects and GSI reports from a target list.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("--model", default="pathf_geophys")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    args = ap.parse_args()
    proc = Path(args.processed)
    mdir = proc / "models" / args.model
    t = gpd.read_file(mdir / "targets.gpkg")
    aoi = gpd.read_file(proc / args.gpkg, layer="aoi").to_crs(t.crs).geometry.iloc[0]
    blocks = gpd.read_file(proc / args.gpkg, layer="exploration_blocks").to_crs(t.crs)
    gblocks = blocks[blocks["has_gold"].astype(bool)].copy()

    # ---- flags and tiers
    ev = t["evidence"].apply(parse_evidence)
    t["geochem_support"] = ev.apply(lambda e: any(n in GEO_POS and z > 0 for n, z in e)) & (t["au_geochem_pct"] >= 50)
    t["ratio_extreme"] = ev.apply(lambda e: any(n in {"eU/eTh", "K/eTh"} and z >= 5 for n, z in e))
    t["low_radiometric"] = ev.apply(lambda e: sum(1 for n, z in e if n in RAD and z <= -1.0) >= 2)
    t["felsic_radiometric"] = ev.apply(lambda e: sum(1 for n, z in e if n in RAD and z >= 1.5) >= 2)
    t["edge"] = t.geometry.distance(aoi.exterior) < 2000
    t["geophysics_only"] = t["au_geochem_pct"] < 50
    flagged = t["ratio_extreme"] | t["low_radiometric"] | t["felsic_radiometric"]
    fresh = t["status"].isin(["new", "near-known"])
    stab = t["stability"].fillna(0)
    t["tier"] = np.select(
        [~fresh, fresh & ~flagged & (stab >= 0.75) & t["geochem_support"], fresh & ~flagged & (stab >= 0.60)],
        ["known", "A", "B"], "C")
    t["flags"] = t.apply(lambda r: ", ".join(f for f in ["ratio_extreme", "low_radiometric", "felsic_radiometric", "edge", "geophysics_only"]
                                             if bool(r[f])), axis=1)

    # ---- prospects: non-known targets within 1 km of each other (known gold ground stays as context)
    cand = t[t["tier"] != "known"]
    buf = cand.geometry.buffer(GROUP_M / 2)
    merged = gpd.GeoDataFrame(geometry=list(buf.union_all().geoms) if buf.union_all().geom_type == "MultiPolygon"
                              else [buf.union_all()], crs=t.crs)
    merged["prospect_ix"] = range(len(merged))
    j = gpd.sjoin(cand[["target", "geometry"]].assign(geometry=cand.geometry.representative_point()),
                  merged, predicate="within")[["target", "prospect_ix"]]
    t = t.merge(j, on="target", how="left")
    tier_rank = {"A": 0, "B": 1, "C": 2, "known": 3}
    rows = []
    for ix, g in t.dropna(subset=["prospect_ix"]).groupby("prospect_ix"):
        best = g.sort_values(["tier", "mean_pct"], key=lambda s: s.map(tier_rank) if s.name == "tier" else -s).iloc[0]
        hull = g.geometry.union_all()
        hits = gblocks[gblocks.intersects(hull.buffer(1000))]
        rows.append({
            "prospect_ix": ix,
            "best_tier": min(g["tier"], key=lambda x: tier_rank[x]),
            "n_targets": len(g), "tiers": " ".join(f"{k}:{v}" for k, v in sorted(g["tier"].value_counts().items())),
            "area_km2": round(g["area_km2"].sum(), 2),
            "status_mix": " ".join(f"{k}:{v}" for k, v in sorted(g["status"].value_counts().items())),
            "best_target": best["target"], "best_mean_pct": best["mean_pct"],
            "mean_stability": round(g["stability"].mean(), 2),
            "worst_run_pct": round(float(g["pct_min_runs"].min()), 2) if "pct_min_runs" in g else np.nan,
            "geochem_support": f"{int(g['geochem_support'].sum())}/{len(g)}",
            "flags": ", ".join(sorted({f for s in g["flags"] for f in s.split(", ") if f})),
            "dist_known_gold_km": g["dist_known_gold_km"].min(),
            "best_evidence": best["evidence"],
            "peak_lon": best["peak_lon"], "peak_lat": best["peak_lat"],
            "gsi_gold_reports": "; ".join(hits.sort_values("area_km2")["report_id"].astype(str).unique()[:6]),
            "geometry": hull.convex_hull if len(g) > 1 else hull,
        })
    pr = gpd.GeoDataFrame(rows, crs=t.crs)
    pr = pr.sort_values(["best_tier", "best_mean_pct"], key=lambda s: s.map(tier_rank) if s.name == "best_tier" else -s)
    pr.insert(0, "prospect", [f"P{i:02d}" for i in range(1, len(pr) + 1)])
    t = t.merge(pr[["prospect_ix", "prospect"]], on="prospect_ix", how="left")

    # ---- reports to pull: gold blocks over Tier A/B prospects, detailed and recent first
    ab = pr[pr["best_tier"].isin(["A", "B"])]
    rep = []
    for _, b in gblocks.iterrows():
        over = ab[ab.intersects(b.geometry.buffer(1000))]
        if over.empty:
            continue
        a = float(b["area_km2"])
        rep.append({"report_id": b["report_id"], "stage": b["stage"], "year_start": b["year_start"],
                    "area_km2": round(a, 1), "scale": "detailed" if a < 50 else ("local" if a < 500 else "regional"),
                    "title": str(b["title"])[:160],
                    "prospects": " ".join(over["prospect"]), "n_prospects": len(over),
                    "folder": f"Exploration_Data/{b['report_id']}/",
                    "report_files": b["report_files"], "table_files": b["table_files"], "gis_files": b["gis_files"]})
    rp = pd.DataFrame(rep)
    if not rp.empty:
        rp["stage_rank"] = rp["stage"].map(STAGE_ORDER).fillna(9)
        rp["scale_rank"] = rp["scale"].map({"detailed": 0, "local": 1, "regional": 2})
        rp = rp.sort_values(["scale_rank", "stage_rank", "n_prospects", "area_km2"],
                            ascending=[True, True, False, True]).drop(columns=["stage_rank", "scale_rank"])

    # ---- write
    unc = [c for c in ("pct_min_runs", "pct_sd_runs") if c in t.columns]
    keep = ["prospect", "target", "tier", "flags", "status", "area_km2", "mean_pct", "stability", *unc, "dist_known_gold_km",
            "au_geochem_pct", "evidence", "gsi_reports", "peak_lon", "peak_lat"]
    tier_sort = t["tier"].map(tier_rank)
    t.assign(_s=tier_sort).sort_values(["_s", "mean_pct"], ascending=[True, False])[keep].to_csv(mdir / "shortlist.csv", index=False)
    pcols = [c for c in pr.columns if c not in ("prospect_ix", "geometry")]
    pr[pcols].to_csv(mdir / "prospects.csv", index=False)
    for p in ("prospects.gpkg", "prospects.geojson"):
        (mdir / p).unlink(missing_ok=True)
    pr[pcols + ["geometry"]].to_file(mdir / "prospects.gpkg", layer="prospects", driver="GPKG")
    ll = pr[pcols + ["geometry"]].to_crs("EPSG:4326")
    ll["geometry"] = ll.geometry.make_valid()
    ll.to_file(mdir / "prospects.geojson", driver="GeoJSON")
    if not rp.empty:
        rp.to_csv(mdir / "reports_to_pull.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 10), dpi=130)
        gpd.GeoSeries([aoi], crs=t.crs).boundary.plot(ax=ax, color="#999", lw=0.6)
        gold = gpd.read_file(proc / args.gpkg, layer="gold_labels")
        gold.plot(ax=ax, color="#2ca25f", markersize=3)
        colours = {"A": "#d7191c", "B": "#fdae61", "C": "#bbbbbb", "known": "#2b8cbe"}
        t[t["tier"] == "known"].plot(ax=ax, facecolor=colours["known"], edgecolor="k", lw=0.3, alpha=0.6)
        for tier in ["C", "B", "A"]:
            sub = pr[pr["best_tier"] == tier]
            if len(sub):
                sub.plot(ax=ax, facecolor=colours[tier], edgecolor="k", lw=0.3, alpha=0.75)
        for _, r in pr[pr["best_tier"].isin(["A", "B"])].iterrows():
            p = r.geometry.representative_point()
            ax.annotate(r["prospect"], (p.x, p.y), fontsize=6.5, xytext=(4, 4), textcoords="offset points", weight="bold")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=f"tier {k}" if k in "ABC" else "known gold ground")
                           for k, c in colours.items()], loc="lower left", fontsize=7)
        ax.set_axis_off()
        ax.set_title(f"V1 prospects ({args.model}); green dots = known gold", fontsize=10)
        fig.tight_layout()
        fig.savefig(mdir / "prospects.png")
    except ImportError:
        pass

    print(f"Targets: {len(t)}  tiers {t['tier'].value_counts().to_dict()}")
    print(f"Flags: ratio_extreme {int(t['ratio_extreme'].sum())}, low_radiometric {int(t['low_radiometric'].sum())}, "
          f"felsic_radiometric {int(t['felsic_radiometric'].sum())}, "
          f"edge {int(t['edge'].sum())}, geophysics_only {int(t['geophysics_only'].sum())}")
    print(f"Prospects: {len(pr)}  best tier {pr['best_tier'].value_counts().to_dict()}\n")
    with pd.option_context("display.width", 230, "display.max_colwidth", 55):
        print(pr[pr["best_tier"].isin(["A", "B"])][["prospect", "best_tier", "n_targets", "area_km2", "mean_stability", "worst_run_pct",
                                                    "geochem_support", "dist_known_gold_km", "flags", "best_evidence",
                                                    "peak_lon", "peak_lat"]].to_string(index=False))
    if not rp.empty:
        print(f"\nGSI reports covering tier A/B prospects: {len(rp)} (top 10)")
        with pd.option_context("display.width", 230, "display.max_colwidth", 70):
            print(rp[["report_id", "scale", "stage", "year_start", "area_km2", "prospects", "title"]].head(12).to_string(index=False))
        print("by scale:", rp["scale"].value_counts().to_dict())
    print(f"\nOutputs in {mdir}: shortlist.csv, prospects.csv, prospects.gpkg/.geojson, reports_to_pull.csv, prospects.png")


if __name__ == "__main__":
    main()
