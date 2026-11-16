#!/usr/bin/env python3
"""
v1_summary.py - one-page summary of the V1 gold model: how it was chosen, how it tests, what it targets.

Run from the repository folder after run_v2e.bat (it is the last step there):
    python v1_summary.py data\\processed --model pf_v1

Reads   models\\<run>\\run.json for pg_ens, pf_ens, pf_gA, pf_gB and the V1 model (cross-validation)
        models\\gsi_eval_foldA.csv, models\\gsi_eval_foldB.csv (blind test on GSI sites held out of training)
        models\\<model>\\step4\\prospects_after_check.csv (else prospects.csv), gsi_prospect_evidence.csv
        models\\pg_ens\\step4\\prospects_after_check.csv (to compare with the earlier shortlist)
Writes  models\\<model>\\V1_SUMMARY.md, and prints it
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}")


def pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def num(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def km(lon1, lat1, lon2, lat2) -> float:
    dx = (lon1 - lon2) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dx, (lat1 - lat2) * 110.57)


def prospects(mdir: Path) -> pd.DataFrame | None:
    for p, tcol in ((mdir / "step4" / "prospects_after_check.csv", "tier_after"), (mdir / "prospects.csv", "best_tier")):
        if p.exists():
            d = pd.read_csv(p)
            return d.rename(columns={tcol: "tier"})
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Summary of the V1 gold model.")
    ap.add_argument("processed")
    ap.add_argument("--model", default="pf_v1")
    ap.add_argument("--compare", default="pg_ens,pf_ens,pf_gA,pf_gB", help="runs shown next to the V1 model")
    args = ap.parse_args()
    proc = Path(args.processed)
    mroot = proc / "models"
    L = [f"# V1 gold prospectivity model: {args.model}", ""]

    # ---- cross-validation
    L += ["## Cross-validation on the 77 known gold labels (leave one region out)", "",
          "| run | GSI labels | top 1% | top 10% | within greenstone, top 10% | within same terrain, top 10% | bag range (greenstone) | AUC |",
          "|---|---|---|---|---|---|---|---|"]
    au_green = au_terr = None
    for r in [x for x in args.compare.split(",") if x] + [args.model]:
        p = mroot / r / "run.json"
        if not p.exists():
            continue
        js = json.loads(p.read_text())
        s, bs = js.get("summary", {}), js.get("bag_spread", {}).get("within_green_top10", {})
        au_green, au_terr = s.get("within_green_pathf_top10", au_green), s.get("terrain_pathf_top10", au_terr)
        g = ("none" if not js.get("gsi_sites") else
             f"{js.get('gsi_positives_added', 0)} gold + {js.get('gsi_negative_cells', 0)} barren"
             + (f", fold {js['gsi_holdout']} held out" if js.get("gsi_holdout", "none") != "none" else ""))
        rng = f"{pct(bs.get('min'))}-{pct(bs.get('max'))}" if bs else ""
        L.append(f"| {r}{' (V1)' if r == args.model else ''} | {g} | {pct(s.get('model_top1'))} | {pct(s.get('model_top10'))} | "
                 f"{pct(s.get('within_green_top10'))} | {pct(s.get('terrain_top10'))} | {rng} | {num(s.get('auc'))} |")
    L += [f"| Au pathfinder index alone | - | - | - | {pct(au_green)} | {pct(au_terr)} | - | - |", "",
          "Random would score 1% / 10% / 10% / 10%. 'Within same terrain' is the fair column: it compares each held-out "
          "label only with ground of the same relief and land cover, so a map cannot score by finding hills.", ""]

    # ---- blind test on held-out GSI folds
    fold_rows = []
    for f in ("A", "B"):
        p = mroot / f"gsi_eval_fold{f}.csv"
        if p.exists():
            d = pd.read_csv(p)
            for _, r in d.iterrows():
                fold_rows.append({"fold": f, **r.to_dict()})
    if fold_rows:
        fr = pd.DataFrame(fold_rows)
        L += ["## Blind test on GSI sites held out of training", "",
              "Each fold's GSI sites (clusters of trench / rock / drill results) were left out when the pf_g model was trained; "
              "pf_ens never saw any GSI site. AUC = chance a gold site ranks above a barren site (0.5 = coin).", "",
              "| fold | map | gold sites | barren sites | AUC | 90% range (clusters) | gold in top 10% | barren in top 10% |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in fr.iterrows():
            ng = int(r.get("n_positive", 0) + r.get("n_anomalous", 0))
            L.append(f"| {r['fold']} | {r['map']} | {ng} | {int(r.get('n_barren', 0))} | {num(r.get('auc_gold_vs_barren'))} | "
                     f"{num(r.get('auc_indep_p05'))}-{num(r.get('auc_indep_p95'))} | {pct(r.get('positive_top10'))} | "
                     f"{pct(r.get('barren_top10'))} |")
        L.append("")

    # ---- prospects
    pv = prospects(mroot / args.model)
    if pv is not None:
        ab = pv[pv["tier"].isin(["A", "B"])]
        L += ["## V1 prospects", "", f"Tier A {int((pv['tier'] == 'A').sum())}, tier B {int((pv['tier'] == 'B').sum())}, "
              f"other {int((~pv['tier'].isin(['A', 'B'])).sum())} (after the automated desk check)." if "statuses" in pv else
              f"Tier A {int((pv['tier'] == 'A').sum())}, tier B {int((pv['tier'] == 'B').sum())}.", ""]
        ev_p = proc / "gsi_prospect_evidence.csv"
        ev = pd.read_csv(ev_p) if ev_p.exists() else None
        old = prospects(mroot / "pg_ens")
        old_ab = old[old["tier"].isin(["A", "B"])] if old is not None else None
        L += ["| prospect | tier | peak lon, lat | km to known gold | GSI evidence | near an earlier pg_ens A/B prospect |",
              "|---|---|---|---|---|---|"]
        for _, p in ab.sort_values(["tier", "prospect"]).iterrows():
            e = ""
            if ev is not None:
                m = ev[ev["prospect"] == p["prospect"]]
                e = str(m["verdict"].iloc[0]) if len(m) else ""
            near = ""
            if old_ab is not None and len(old_ab):
                d = min(km(p["peak_lon"], p["peak_lat"], o["peak_lon"], o["peak_lat"]) for _, o in old_ab.iterrows())
                near = f"yes ({d:.1f} km)" if d <= 2 else "no"
            dk = p.get("dist_known_gold_km", float("nan"))
            L.append(f"| {p['prospect']} | {p['tier']} | {p['peak_lon']:.4f}, {p['peak_lat']:.4f} | "
                     f"{'' if pd.isna(dk) else f'{dk:.1f}'} | {e} | {near} |")
        if old_ab is not None and len(old_ab) and len(ab):
            kept = sum(any(km(p["peak_lon"], p["peak_lat"], o["peak_lon"], o["peak_lat"]) <= 2 for _, o in ab.iterrows())
                       for _, p in old_ab.iterrows())
            L += ["", f"Of the {len(old_ab)} earlier pg_ens tier A/B prospects, {kept} have a V1 A/B prospect within 2 km."]
        L.append("")

    L += ["## Known limits", "",
          "- The model ranks only ground within 1 km of mapped greenstone; gold at granite contacts (SRO-39129-2021, "
          "15.1 g/t, 2.8 km off the belt) is outside it.",
          "- No map, trained or not, separated GSI gold follow-up sites from GSI barren follow-up sites well: the data "
          "(regional geochemistry on a 1-4 km sampling grid) picks prospective belts, not the winning target inside them.",
          "- GSI sites cluster in a few blocks; differences under ~0.1 AUC are noise.", ""]
    text = "\n".join(L)
    (mroot / args.model).mkdir(parents=True, exist_ok=True)
    (mroot / args.model / "V1_SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Written {mroot / args.model / 'V1_SUMMARY.md'}")


if __name__ == "__main__":
    main()
