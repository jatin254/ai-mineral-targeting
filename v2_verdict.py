#!/usr/bin/env python3
"""
v2_verdict.py - applies the V2 pass bar (fixed in the V1 write-up before any V2 run) to every contender.

Run from the repository folder, last step of run_v3.bat:
    python v2_verdict.py data\\processed

Pass bar, all three:
  1. blind test: AUC on GSI gold vs barren sites (independent sites, none used in training) at least 0.10 above
     the Au pathfinder index alone on the same sites;
  2. and above the index in at least 90% of the resamples of whole site clusters;
  3. cross-validation within the same terrain (top 10%) no lower than the index.
If nothing passes: keep V1 as the screening map and stop tuning the gold model.

Reads   models\\gsi_eval_v2.csv, models\\gsi_eval_v2_belt3.5km.csv (gsi_eval.py), models\\<run>\\run.json
Writes  models\\V2_VERDICT.md, and prints it
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


import pandas as pd

CONTENDERS = [  # run, what it tests, blind-test file tag
    ("pf_ens", "V1: trees on the 5 geochemistry layers", "_v2"),
    ("cnn_pf", "CNN on the same 5 layers (is the algorithm the gap?)", "_v2"),
    ("v2_struct", "trees + shear zones and other structure", "_v2"),
    ("v2_all", "trees + structure + soil + stream geochemistry", "_v2"),
    ("cnn_all", "CNN on all V2 layers", "_v2"),
    ("v2_b35", "trees on all V2 layers, 3.5 km belt (off-belt gold)", "_v2_belt3.5km"),
]
MIN_GAIN, MIN_SHARE = 0.10, 0.90


def f2(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def pc(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def paired_share(proc: Path, a: str, b: str, n_boot: int = 2000) -> float | None:
    """Share of cluster resamples in which map a's independent AUC beats map b's (same resamples for both)."""
    p = proc / "gsi_eval_sites_v2.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p)
    if f"pct_{a}" not in s or f"pct_{b}" not in s:
        return None
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gsi_eval import bootstrap
    on = s["on_belt"].astype(bool).to_numpy() & s["independent"].astype(bool).to_numpy()
    g = on & s["label"].isin(["positive", "anomalous"]).to_numpy()
    br = on & s["label"].isin(["barren_strong", "barren"]).to_numpy()
    sp = {b: s[f"pct_{b}"].to_numpy(float), a: s[f"pct_{a}"].to_numpy(float)}
    res = bootstrap(sp, g, br, s["cluster"].to_numpy(), n_boot, b)
    v = res[a]["share_beats_au_index"]          # here: share beating map b
    return None if v != v else float(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the V2 pass bar.")
    ap.add_argument("processed")
    args = ap.parse_args()
    mroot = Path(args.processed) / "models"
    evals = {}
    for tag in {t for _, _, t in CONTENDERS}:
        p = mroot / f"gsi_eval{tag}.csv"
        if p.exists():
            evals[tag] = pd.read_csv(p).set_index("map")
    L = ["# V2 verdict", "",
         f"Pass bar (fixed before the runs): blind-test AUC at least {MIN_GAIN:.2f} above the Au index, above it in at least "
         f"{MIN_SHARE:.0%} of cluster resamples, and cross-validation within the same terrain no lower than the index.", "",
         "| run | what it tests | CV same terrain, model / index | blind AUC, model / index | gain | beats index in | verdict |",
         "|---|---|---|---|---|---|---|"]
    passed, rows = [], {}
    for run, what, tag in CONTENDERS:
        rj = mroot / run / "run.json"
        ev = evals.get(tag)
        if not rj.exists() or ev is None or run not in ev.index:
            L.append(f"| {run} | {what} | not run | | | | - |")
            continue
        s = json.loads(rj.read_text()).get("summary", {})
        t_m, t_i = s.get("terrain_top10"), s.get("terrain_pathf_top10")
        a_m, a_i = ev.loc[run, "auc_independent"], ev.loc["Au index alone", "auc_independent"]
        share = ev.loc[run, "share_beats_au_index"]
        gain = a_m - a_i
        why = []
        if not gain >= MIN_GAIN:
            why.append(f"gain {gain:+.3f} < {MIN_GAIN:.2f}")
        if not share >= MIN_SHARE:
            why.append(f"beats index in {share:.0%} < {MIN_SHARE:.0%}")
        if t_m is None or t_i is None or not t_m >= t_i - 0.005:   # half a point: equal as printed counts as equal
            why.append("terrain CV below index")
        ok = not why
        if ok:
            passed.append(run)
        rows[run] = {"a": a_m, "t": t_m}
        L.append(f"| {run} | {what} | {pc(t_m)} / {pc(t_i)} | {f2(a_m)} / {f2(a_i)} | {gain:+.2f} | {pc(share)} | "
                 f"{'PASS' if ok else 'fail: ' + '; '.join(why)} |")
    L.append("")
    if "pf_ens" in rows and "cnn_pf" in rows:
        d = rows["cnn_pf"]["a"] - rows["pf_ens"]["a"]
        share = paired_share(Path(args.processed), "cnn_pf", "pf_ens")
        clear = d >= MIN_GAIN and share is not None and share >= MIN_SHARE
        L.append(f"Algorithm test (same 5 inputs): CNN blind AUC {f2(rows['cnn_pf']['a'])} vs trees {f2(rows['pf_ens']['a'])} "
                 f"({d:+.2f}; CNN ahead in {pc(share)} of cluster resamples). "
                 + ("The CNN is clearly better: the algorithm matters here." if clear else
                    "Not a clear difference: on these inputs the algorithm is not the gap."))
        L.append("")
    if passed:
        L.append(f"**Verdict: {', '.join(passed)} passed.** Rebuild targets, shortlist and desk check from "
                 f"{passed[0]} and write V2.")
    else:
        L.append("**Verdict: no contender passed.** Keep V1 (pf_ens) as the screening map, stop tuning the gold model, "
                 "and treat the data, not the algorithm, as the limit.")
    text = "\n".join(L)
    (mroot / "V2_VERDICT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten {mroot / 'V2_VERDICT.md'}")


if __name__ == "__main__":
    main()
