#!/usr/bin/env python3
"""
compare_runs.py - side-by-side held-out results of several train_v1 runs.

    python compare_runs.py data/processed pg_ens pg_tm05 pg_tm10

Columns (all from leave-one-region-out, weighted by positives; random in brackets):
  top1 / top10       share of held-out gold in the top 1% / 10% of the ground  (1% / 10%)
  green10            top 10% within greenstone: model vs Au pathfinder index alone  (10%)
  terrain10          top 10% within the same terrain bin: model vs Au index  (10%)
                     the number that says whether the model knows more than "exposed rock"
  blend              the 50/50 rank blend of model and Au index: top1 / green10 / terrain10
  AUC                held-out labels vs background
  top1% terrain      median relief and share mostly scrub of the model's top 1% of belt ground
  bag range          what single bags would have scored for green10 (spread = instability)
Writes data/processed/models/compare_runs.csv as well.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def pct(v):
    return "  n/a" if v is None or v != v else f"{v:5.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare train_v1 runs.")
    ap.add_argument("processed")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    mroot = Path(args.processed) / "models"
    rows = []
    for r in args.runs:
        p = mroot / r / "run.json"
        if not p.exists():
            print(f"{r}: no run.json, skipped")
            continue
        js = json.loads(p.read_text())
        s = js.get("summary", {})
        tm = s.get("terrain_mix", {}).get("top1pct", {})
        bs = js.get("bag_spread", {}).get("within_green_top10", {})
        rows.append({"run": r, "bags": js.get("bags", 1), "match_terrain": js.get("match_terrain", 0.0),
                     "top1": s.get("model_top1"), "top10": s.get("model_top10"),
                     "green10_model": s.get("within_green_top10"), "green10_au_index": s.get("within_green_pathf_top10"),
                     "terrain10_model": s.get("terrain_top10"), "terrain10_au_index": s.get("terrain_pathf_top10"),
                     "auc": s.get("auc"), "top1_relief_m": tm.get("median_relief_m"),
                     "top1_mostly_scrub": tm.get("share_mostly_scrub"),
                     "bag_green10_min": bs.get("min"), "bag_green10_max": bs.get("max"),
                     "blend_top1": s.get("blend_top1"), "blend_green10": s.get("blend_green10"),
                     "blend_terrain10": s.get("blend_terrain10"),
                     "terrain_correct": ",".join(js.get("terrain_correct", [])), "blend_au": js.get("blend_au", 0.0)})
    if not rows:
        sys.exit("nothing to compare")
    head = (f"{'run':12}{'top1':>6}{'green10':>14}{'terrain10':>14}{'blend top1/green/terr':>23}"
            f"{'AUC':>6}{'top1% terrain':>15}{'bag range':>11}")
    print(head + "\n" + " " * 18 + f"{'model/Au idx':>14}{'model/Au idx':>14}")
    for x in rows:
        terr = (f"{x['top1_relief_m']:.0f} m/{x['top1_mostly_scrub']:.0%}" if x["top1_relief_m"] is not None else "n/a")
        rng = (f"{x['bag_green10_min']:.0%}-{x['bag_green10_max']:.0%}" if x["bag_green10_min"] is not None else "")
        bl = f"{pct(x['blend_top1']).strip()}/{pct(x['blend_green10']).strip()}/{pct(x['blend_terrain10']).strip()}"
        auc = "n/a" if x["auc"] is None else f"{x['auc']:.2f}"
        print(f"{x['run']:12}{pct(x['top1']):>6}"
              f"{pct(x['green10_model']):>7}/{pct(x['green10_au_index']):>6}"
              f"{pct(x['terrain10_model']):>7}/{pct(x['terrain10_au_index']):>6}"
              f"{bl:>23}{auc:>6}{terr:>15}{rng:>11}")
    print("random: top1 1%, green10 and terrain10 10%")
    with open(mroot / "compare_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {mroot / 'compare_runs.csv'}")


if __name__ == "__main__":
    main()
