#!/usr/bin/env python3
"""
gsi_labels.py - turn GSI report tables into gold sample points, V2 label sites and per-prospect evidence.

Run from the repository folder after downloading reports into Reports\\<report_id>\\ :
    python gsi_labels.py data\\processed --reports-dir Reports --model pg_ens

Reads   every .xlsx/.xls/.csv under Reports\\<report_id>\\ (zips and rars are unpacked next to them),
        data\\processed\\gsi_manual_points.csv   samples typed in from PDF-only reports (optional)
        data\\processed\\gsi_findings_*.csv      report summaries to merge into report_findings.csv
        ka_ap_v1.gpkg gold_labels, models\\<model>\\prospects (desk-checked tiers if available)
Writes  data\\processed\\gsi_samples.csv / .gpkg   one row per sample (or per trench) with lon, lat,
                                                  gold in ppb, sample type, report, sheet
        data\\processed\\gsi_label_sites.csv / .gpkg  samples merged into sites (250 m):
                 positive        a rock, channel, trench, pit, drill or underground sample >= 0.5 g/t
                 anomalous       best rock sample 0.1-0.5 g/t, or soil/stream >= 100 ppb (not a label)
                 barren_strong   >= 3 trench/pit/drill/channel samples, all < 0.1 g/t
                 barren          >= 3 rock samples (bedrock, chip) all < 0.1 g/t
                 barren_weak     soil / stream sediment only, all < 50 ppb
        data\\processed\\gsi_prospect_evidence.csv  for every tier A/B prospect: nearest gold found,
                 barren testing nearby, samples within 1 km of the peak, and a one-line verdict
        data\\processed\\report_findings.csv          filled in from gsi_findings_*.csv where still blank
How tables are read: the header row and a units row under it are found automatically; gold columns
(Au / Gold, units ppb, ppm, g/t) and coordinates (decimal degrees, or D-M-S columns) are detected;
sample type comes from the sheet title (bed rock, stream sediment, soil, trench, channel, pit,
borehole, underground) or else from the sample-number prefix. Check-analysis, mineral-chemistry and
statistics sheets are skipped. Trench sheets that list only From/To intervals are matched to the
trench coordinate sheet by trench order and sample counts. Tables without coordinates (old reports
that give only village names) are listed but cannot become points.
GIS files: unzip nothing yourself; put each report's GIS_FILES .gdb.zip / .zip in its folder (zips inside
zips, rar and 7z are unpacked too). Point, line and polygon layers, and GIS tables with lon/lat or UTM
columns, are read and matched to the tables by sample, trench or borehole number (whichever field
matches best), so trench HNT-1 gives its position to samples HNT-1/1, HNT-1/2 ...; layers that carry
their own gold field become samples directly (unit from the field name, else guessed from the values).
Layers without a projection file are taken as degrees or UTM 43N from their extent; features outside
the study block are ignored. Every layer is listed in gsi_gis_layers.csv: which field matched, how many
samples it located, and, where nothing matched, example ids from the layer and from the tables.
Gold >= 0.5 g/t still without a position is listed in gsi_unlocated_gold.csv. Gold written into GIS text
is read too: REMARKS such as 'Gold value 3.4 g/t' or '0.38/1.5m' (g/t over width, only when that is the
whole remark), and results appended to sample numbers ('PH/PT-37 0.79/2.5 M'); other commodities skipped.
gsi_gis_samples.csv holds every feature of the sample-type layers with all attributes, for checking.
Checks: decimal commas ('0,20'), missing decimal points ('021'), ppb values under a 'ppm' heading
(detected from '<25'-type detection limits) are handled; values above 500 g/t, or above 100 g/t in a
column whose typical value is below 1 g/t, are left out and listed as 'suspect value' for checking.
Also writes data\\processed\\gsi_report_summary.csv (best result per report, with or without coordinates)
and lists, per prospect, the downloaded reports whose exploration block overlaps it.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    from pyproj import Transformer
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsi_reports import header_row, safe_extract  # noqa: E402

POS_PPB, ANOM_PPB, BARREN_PPB, WEAK_PPB = 500.0, 100.0, 100.0, 50.0
SITE_M = 250.0
ROCK = {"bedrock", "rock_chip", "channel", "trench", "pit", "drill", "underground"}
STRONG = {"trench", "pit", "drill", "channel"}
TYPE_WORDS = [("stream sediment", "stream"), ("stream", "stream"), ("soil", "soil"), ("bed rock", "bedrock"),
              ("bedrock", "bedrock"), ("rock chip", "rock_chip"), ("trench", "trench"), ("channel", "channel"),
              ("pit", "pit"), ("bore hole", "drill"), ("borehole", "drill"), ("drill", "drill"), ("core", "drill"),
              ("underground", "underground"), ("mine", "underground")]
PREFIX = [(r"SSS", "stream"), (r"SS", "soil"), (r"BR", "bedrock"), (r"TS|TR|T\d", "trench"), (r"PT|PIT", "pit"),
          (r"CH|CS", "channel"), (r"DD|BH|OD", "drill"), (r"RC|GR", "rock_chip")]
SKIP_SHEET = re.compile(r"check|duplicate|epma|mineral chem|fluid inclusion|statistic|parameter|locality|petrograph", re.I)
SKIP_COLS = re.compile(r"^(formula|total|mineral|parameter|dataset/point|inclusion)$", re.I)
AU_LABEL = re.compile(r"(^|[\s_(/])(au|gold)([\s_)/]|$)|gold\s*assay|^au\b", re.I)
NOT_AU = re.compile(r"radio|pathfinder|weighted|index", re.I)
LAT_LABEL, LON_LABEL = re.compile(r"\blat", re.I), re.compile(r"\blong|\blon\b", re.I)
SAMPLE_LABEL = re.compile(r"sample|^s\.?\s*no|borehole|trench\s*no|^nos?$", re.I)
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


# ----------------------------------------------------------------------------- parsing helpers
def unit_factor(text: str) -> float | None:
    t = text.lower().replace(" ", "")
    if "ppb" in t:
        return 1.0
    if "ppm" in t:
        return 1000.0
    if re.search(r"g/t|gm/t|gram|gms|qms|g/m\.?t|gm/mt|grams/t|gpt", t):
        return 1000.0
    return None


def parse_value(s) -> tuple[float | None, bool]:
    """(value, below_detection). '<25' -> (0, True); 'ND', 'Traces', '0.1*' -> (0, True)."""
    if s is None:
        return None, False
    t = str(s).strip()
    if t.lower() in ("", "nan", "none", "-", "--", "na", "n.a."):
        return None, False
    if t.startswith("<") or t.lower() in ("nd", "n.d.", "traces", "trace", "tr", "bdl", "bld") or t.endswith("*"):
        return 0.0, True
    if re.fullmatch(r"0,\d+|\d+,\d{1,2}", t):     # decimal comma: '0,20' -> 0.20
        t = t.replace(",", ".")
    t = t.replace(",", "")                          # thousands separator: '1,250' -> 1250
    if re.fullmatch(r"0\d+", t):                     # missing decimal point: '021' -> 0.21
        t = "0." + t[1:]
    m = re.fullmatch(r"[~>]?\s*(\d+(\.\d+)?|\.\d+)", t)
    return (float(m.group(1)), False) if m else (None, False)


def to_float(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def is_unit_row(vals: list) -> bool:
    cells = [str(v).strip() for v in vals if str(v).strip() not in ("", "nan", "None")]
    if not cells:
        return False
    numeric = sum(to_float(c) is not None for c in cells)
    return numeric <= len(cells) * 0.2


def sheet_title(raw: pd.DataFrame, h: int) -> str:
    texts = []
    for i in range(min(h, 4)):
        texts += [str(v) for v in raw.iloc[i].tolist() if str(v).strip() not in ("", "nan", "None")]
    return " ".join(texts)[:300]


def type_from(text: str, sample: str) -> str:
    low = text.lower()
    for w, t in TYPE_WORDS:
        if w in low:
            return t
    raw = re.sub(r"[^A-Z0-9]", "", str(sample).upper())
    stripped = re.sub(r"^(KH|KK|KC|[A-Z]{2})(?=[A-Z])", "", raw)   # a 2-letter area code (KKSS-4 -> SS, CKBRS -> BRS)
    for s in dict.fromkeys((stripped, raw)):                      # then as written ('BRS-178' has no area code)
        for pat, t in PREFIX:
            if re.match(pat, s):
                return t
    return "unknown"


def type_from_layer(name: str) -> str:
    """Sample type from a shapefile name such as 'XYBRS_gold1.shp' (BRS = bed rock) or 'XYCKCS_Gold' (CS = channel)."""
    for tok in re.split(r"[_\W]+", re.sub(r"\.\w+$", "", name.split(":")[-1])):
        tok = re.sub(r"^XY", "", tok.upper())
        if 2 <= len(tok) <= 6 and tok.isalpha() and not re.fullmatch(r"GOLD\w*|AU\w*|FINAL|REVIEW\w*|SINGLE", tok):
            t = type_from("", tok)
            if t != "unknown":
                return t
    return "unknown"


def dedupe_gis(samples: list[dict]) -> tuple[list[dict], int]:
    """GIS zips often hold the same shapefile several times (and review copies of it): keep one row per
    report + position (~10 m) + value, preferring a row with a sample number and a known type; a sample
    number seen at the same position with two values (a re-assay) keeps the higher value."""
    gis = [r for r in samples if str(r.get("sheet", "")).startswith("GIS") and r.get("lat") is not None]
    rest = [r for r in samples if not (str(r.get("sheet", "")).startswith("GIS") and r.get("lat") is not None)]
    best = {}
    for r in gis:
        k = (r["report_id"], round(r["lon"], 4), round(r["lat"], 4), r["au_ppb"])
        score = (has(r.get("sample_no")) and not str(r["sample_no"]).count("#"), r.get("sample_type") != "unknown")
        if k not in best or score > best[k][0]:
            best[k] = (score, r)
    kept = [v[1] for v in best.values()]
    by_id = {}
    for r in kept:
        sid = norm_id(r["sample_no"]) if has(r.get("sample_no")) and "#" not in str(r["sample_no"]) else None
        k = (r["report_id"], round(r["lon"], 4), round(r["lat"], 4), sid) if sid else id(r)
        if k not in by_id or r["au_ppb"] > by_id[k]["au_ppb"]:
            by_id[k] = r
    kept = list(by_id.values())
    return rest + kept, len(gis) - len(kept)


def pick_coord(df: pd.DataFrame, labels: list[str], pat: re.Pattern, lo: float, hi: float):
    best, best_n = None, 0
    for c, lab in zip(df.columns, labels):
        if not pat.search(lab):
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        ok = v.between(lo, hi) & (v % 1 != 0)
        if ok.sum() > best_n:
            best, best_n = c, int(ok.sum())
    return best


def parse_dms(s):
    """'13° 55' 26.74" N', '14 47 33.8', '14o 49’ 10’’', '76°2'21.4"' -> decimal degrees."""
    t = str(s).strip()
    if t in ("", "nan", "None"):
        return None
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    if not 2 <= len(nums) <= 3:
        return None
    d, m = float(nums[0]), float(nums[1])
    sec = float(nums[2]) if len(nums) == 3 else 0.0
    if m >= 60 or sec >= 60:
        return None
    v = d + m / 60 + sec / 3600
    return -v if re.search(r"[SW]\s*$", t, re.I) else v


def coord_series(df: pd.DataFrame, labels: list[str], pat: re.Pattern, lo: float, hi: float, which: str):
    """Decimal degrees column, else a degrees-minutes-seconds text column, else D/M/S number columns."""
    c = pick_coord(df, labels, pat, lo, hi)
    if c is not None:
        return pd.to_numeric(df[c], errors="coerce")
    best, best_n = None, 0
    for c, lab in zip(df.columns, labels):
        if pat.search(lab):
            v = df[c].map(parse_dms)
            n_ok = int(pd.to_numeric(v, errors="coerce").between(lo, hi).sum())
            if n_ok > best_n:
                best, best_n = pd.to_numeric(v, errors="coerce"), n_ok
    if best is not None and best_n >= 3:
        return best.where(best.between(lo, hi))
    return dms(df, labels, which)


def dms(df: pd.DataFrame, labels: list[str], which: str):
    """D/M/S (lat) or D1/M1/S1 (lon) triplets -> decimal degrees."""
    want = ("D", "M", "S") if which == "lat" else ("D1", "M1", "S1")
    idx = {}
    for c, lab in zip(df.columns, labels):
        last = lab.split()[-1] if lab.split() else ""
        if last in want and last not in idx:
            idx[last] = c
    if len(idx) < 3:
        return None
    d, m, s = (pd.to_numeric(df[idx[k]], errors="coerce") for k in want)
    return d + m / 60 + s.fillna(0) / 3600


# ----------------------------------------------------------------------------- reading reports
def read_sheet(report: str, path: Path, sheet: str, raw: pd.DataFrame, log: list) -> list[dict]:
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty or len(raw) < 2:
        return []
    h = header_row(raw)
    title = sheet_title(raw, h)
    head = [str(v).strip() if str(v) not in ("nan", "None") else "" for v in raw.iloc[h].tolist()]
    body_start = h + 1
    sub = raw.iloc[h + 1].tolist() if len(raw) > h + 1 else []
    if sub and is_unit_row(sub):
        sub = [str(v).strip() if str(v) not in ("nan", "None") else "" for v in sub]
        filled, last = [], ""
        for a, b in zip(head, sub):   # a merged parent header ("Assay in ppm") covers the sub-columns under it
            last = a if a else (last if b else "")
            filled.append(a or (last if b else ""))
        labels = [f"{a} {b}".strip() for a, b in zip(filled, sub)]
        body_start = h + 2
    else:
        labels = head
    body = raw.iloc[body_start:].reset_index(drop=True)
    info = {"report_id": report, "file": path.name, "sheet": sheet, "title": title[:120], "rows": len(body)}
    if SKIP_SHEET.search(sheet) or SKIP_SHEET.search(title) or any(SKIP_COLS.match(l) for l in labels) \
            or any(re.match(r"^check", l, re.I) for l in labels):
        info["status"] = "skipped (check / mineral chemistry / statistics / index)"
        log.append(info)
        return []
    au_cols = [(c, l) for c, l in zip(body.columns, labels) if AU_LABEL.search(l) and not NOT_AU.search(l)]
    if not au_cols and re.search(r"\bgold\b", title, re.I):   # "results of trench samples for Gold" + an "Assay" column
        au_cols = [(c, l) for c, l in zip(body.columns, labels) if re.search(r"assay", l, re.I) and not NOT_AU.search(l)]
    if not au_cols:
        info["status"] = "no gold column"
        log.append(info)
        return []
    def score(cl):
        c, l = cl
        vals = body[c].map(lambda v: parse_value(v)[0])
        return (unit_factor(l) is not None, int(vals.notna().sum()))
    au_c, au_l = max(au_cols, key=score)
    fac = unit_factor(au_l)
    unit_note = ""
    if fac is None:   # a units cell elsewhere in the sheet ("All values in ppm", "* gold and silver g/m.tonne")
        notes = " ".join(str(v) for v in raw.values.ravel() if isinstance(v, str))
        fac = unit_factor(title) or (1000.0 if re.search(r"gold[^.]{0,30}(g/m|gram|g/t)", notes, re.I) else None)
    vals_s = body[au_c].astype(str).str.strip()
    det_int = vals_s.str.fullmatch(r"<\s*\d{2,}").sum()      # '<25', '<20': detection limits only ppb tables have
    if fac is None and vals_s.str.fullmatch(r"<\s*25").any():
        fac, unit_note = 1.0, " (assumed from the <25 detection limit)"   # GSI's usual gold detection limit, in ppb
    elif fac == 1000.0 and det_int >= max(3, 0.3 * vals_s.ne("nan").sum()):
        fac, unit_note = 1.0, " (header says ppm or g/t, but <25-type detection limits mean ppb)"
    lat = coord_series(body, labels, LAT_LABEL, 5, 40, "lat")
    lon = coord_series(body, labels, LON_LABEL, 60, 100, "lon")
    s_cols = [c for c, l in zip(body.columns, labels) if SAMPLE_LABEL.search(l) and not re.search(r"type|length", l, re.I)]
    s_c = s_cols[0] if s_cols else None
    hole_c = next((c for c, l in zip(body.columns, labels) if c != s_c and
                   re.search(r"trench|bore\s*hole|drill\s*hole|bh\s*no|pit\s*no", l, re.I)), None)
    frm = next((c for c, l in zip(body.columns, labels) if re.search(r"\bfrom\b", l, re.I)), None)
    to = next((c for c, l in zip(body.columns, labels) if re.search(r"\bto\b", l, re.I)), None)
    info.update({"gold_column": au_l, "unit": {1.0: "ppb", 1000.0: "ppm or g/t"}.get(fac, "unknown") + unit_note,
                 "has_coords": lat is not None and lon is not None and lat.notna().any()})
    recs = []
    pos_vals = [x for x in (parse_value(v)[0] for v in body[au_c]) if x]
    col_median = float(np.median(pos_vals)) * (fac or 0) if pos_vals else 0.0
    for i, row in body.iterrows():
        v, bdl = parse_value(row[au_c])
        if v is None:
            continue
        sample = str(row[s_c]).strip() if s_c is not None and str(row[s_c]) != "nan" else ""
        if hole_c is not None and str(row[hole_c]) not in ("nan", ""):
            hole = str(row[hole_c]).strip()
            if not norm_id(sample).startswith(norm_id(hole)):
                sample = f"{hole}/{sample}" if sample else hole   # 'GNT-6' + '1' -> 'GNT-6/1'
        # > 500 g/t, or > 100 g/t in a column whose typical value is < 1 g/t: almost surely a typing or
        # unit error in the source table (e.g. ppb values pasted into a ppm column)
        if fac is not None and (v * fac > 500_000 or (v * fac > 100_000 and col_median < 1000)):
            info["suspect"] = info.get("suspect", 0) + 1
            log.append({"report_id": report, "file": path.name, "sheet": sheet,
                        "status": f"suspect value left out: {sample} = {row[au_c]} ({au_l}); check the report"})
            continue
        rec = {"report_id": report, "file": path.name, "sheet": sheet, "sample_no": sample,
               "sample_type": type_from(title + " " + sheet, sample), "au_raw": str(row[au_c]).strip(),
               "au_ppb": None if fac is None else v * fac, "below_detection": bdl,
               "lon": float(lon.iloc[i]) if lon is not None and pd.notna(lon.iloc[i]) else None,
               "lat": float(lat.iloc[i]) if lat is not None and pd.notna(lat.iloc[i]) else None,
               "from_m": to_float(row[frm]) if frm is not None else None,
               "to_m": to_float(row[to]) if to is not None else None, "source": "table"}
        recs.append(rec)
    info["samples"] = len(recs)
    info["with_coords"] = sum(r["lat"] is not None for r in recs)
    info["status"] = "read" if info["with_coords"] else "read (no coordinates)"
    log.append(info)
    return recs


def read_coords_sheet(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Trench coordinate sheets have no gold column, so read_sheet stops early; read them here."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if len(raw) < 3:
        return None
    h = header_row(raw)
    labels = [str(v).strip() if str(v) not in ("nan", "None") else "" for v in raw.iloc[h].tolist()]
    body = raw.iloc[h + 1:].reset_index(drop=True)
    cnt_c = next((c for c, l in zip(body.columns, labels) if re.search(r"no\.?\s*of\s*samples", l, re.I)), None)
    tid_c = next((c for c, l in zip(body.columns, labels) if re.search(r"trench\s*no", l, re.I)), None)
    if cnt_c is None or tid_c is None:
        return None
    lat = coord_series(body, labels, LAT_LABEL, 5, 40, "lat")
    lon = coord_series(body, labels, LON_LABEL, 60, 100, "lon")
    if lat is None or lon is None:
        return None
    t = pd.DataFrame({"trench": body[tid_c].astype(str).str.strip(), "n": pd.to_numeric(body[cnt_c], errors="coerce"),
                      "lat": lat, "lon": lon}).dropna()
    return t[t["n"] > 0].reset_index(drop=True)


def link_trenches(recs: list[dict], trenches: pd.DataFrame, log: list, report: str) -> list[dict]:
    """Interval rows without coordinates + a trench sheet whose sample counts add up -> one row per trench."""
    out = []
    by_sheet = {}
    for r in recs:
        if r["lat"] is None and r["from_m"] is not None:
            by_sheet.setdefault((r["file"], r["sheet"]), []).append(r)
    for (f, sh), rows in by_sheet.items():
        if int(trenches["n"].sum()) != len(rows):
            log.append({"report_id": report, "file": f, "sheet": sh,
                        "status": f"trench intervals not matched ({len(rows)} rows vs {int(trenches['n'].sum())} in the trench list)"})
            continue
        i = 0
        for _, t in trenches.iterrows():
            seg = rows[i:i + int(t["n"])]
            i += int(t["n"])
            lens = [max((s["to_m"] or 0) - (s["from_m"] or 0), 0) for s in seg]
            vals = [s["au_ppb"] or 0 for s in seg]
            wavg = sum(v * l for v, l in zip(vals, lens)) / max(sum(lens), 1e-9)
            out.append({"report_id": report, "file": f, "sheet": sh, "sample_no": t["trench"], "sample_type": "trench",
                        "au_raw": f"max {max(vals) / 1000:.3f} g/t, {wavg / 1000:.3f} g/t over {sum(lens):.1f} m ({len(seg)} samples)",
                        "au_ppb": max(vals), "below_detection": max(vals) == 0, "lon": float(t["lon"]), "lat": float(t["lat"]),
                        "from_m": None, "to_m": None, "source": "trench intervals matched to trench coordinates",
                        "n_samples": len(seg)})
        log.append({"report_id": report, "file": f, "sheet": sh, "status": f"{len(trenches)} trenches matched by order and sample counts"})
    return out


ARCHIVE_EXT = (".zip", ".rar", ".7z")


def fix_gdb_dirs(dest: Path) -> None:
    """A .gdb.zip sometimes holds the a0000000x.gdbtable files without their .gdb folder; GDAL only
    opens a folder named *.gdb, so give them one (inside our own _extracted folder only)."""
    for d in sorted({p.parent for p in dest.rglob("*.gdbtable")}, key=lambda p: -len(p.parts)):
        if d.suffix.lower() == ".gdb":
            continue
        if d == dest:
            new = dest / (re.sub(r"\.gdb$", "", dest.name, flags=re.I) + ".gdb")
            new.mkdir(exist_ok=True)
            for c in list(dest.iterdir()):
                if c.is_file() and re.match(r"(a[0-9a-f]{8}\.|gdb$|timestamps$)", c.name, re.I):
                    shutil.move(str(c), str(new / c.name))
        else:
            d.rename(d.with_name(d.name + ".gdb"))


def unpack_archives(rep_dir: Path) -> list[str]:
    """Unpack zip / rar / 7z files, including archives inside archives (up to 4 levels).
    Each goes to <its folder>/_extracted/<name>. Returns one note per archive."""
    notes, seen_arch = [], set()
    for _ in range(4):
        new = 0
        for f in sorted(rep_dir.rglob("*")):
            low = f.name.lower()
            if not f.is_file() or not low.endswith(ARCHIVE_EXT):
                continue
            dest = f.parent / "_extracted" / re.sub(r"\.gdb$", "", f.stem, flags=re.I)
            if dest.exists():   # unpacked by an earlier run: still repair a folder-less gdb, and say so once
                if f.name not in seen_arch:
                    fix_gdb_dirs(dest)
                    notes.append(f"{f.name}: unpacked ({sum(1 for p in dest.rglob('*') if p.is_file())} files, earlier run)")
                    seen_arch.add(f.name)
                continue
            try:
                if low.endswith(".zip"):
                    safe_extract(f, dest)
                elif shutil.which("tar"):   # Windows 10+ tar (bsdtar) reads rar and 7z
                    dest.mkdir(parents=True, exist_ok=True)
                    r = subprocess.run(["tar", "-xf", str(f), "-C", str(dest)], capture_output=True, text=True, timeout=300)
                    if r.returncode != 0:
                        shutil.rmtree(dest, ignore_errors=True)
                        notes.append(f"{f.name}: could not unpack ({' '.join(r.stderr.split())[:70]}); open it with 7-Zip")
                        continue
                else:
                    notes.append(f"{f.name}: no tool to unpack it here; open it with 7-Zip")
                    continue
                fix_gdb_dirs(dest)
                n_files = sum(1 for p in dest.rglob("*") if p.is_file())
                notes.append(f"{f.name}: unpacked ({n_files} files)")
                seen_arch.add(f.name)
                new += 1
            except Exception as ex:
                notes.append(f"{f.name}: could not unpack ({type(ex).__name__}: {str(ex)[:60]})")
        if not new:
            break
    return notes


def read_reports(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    samples, log, gis_log = [], [], []
    for rep_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rep = rep_dir.name
        notes = unpack_archives(rep_dir)
        files, seen = [], set()
        for f in sorted(rep_dir.rglob("*")):
            if f.suffix.lower() in (".xlsx", ".xls", ".csv") and not f.name.startswith("~$") and f.is_file():
                h = hashlib.md5(f.read_bytes()).hexdigest()
                if h not in seen:   # the same table downloaded loose and inside a zip is read once
                    seen.add(h)
                    files.append(f)
        recs_rep, trench_lists = [], []
        for f in files:
            try:
                sheets = ({"csv": pd.read_csv(f, header=None, dtype=str, encoding_errors="replace")} if f.suffix.lower() == ".csv"
                          else pd.read_excel(f, sheet_name=None, header=None, dtype=str))
            except Exception as ex:
                log.append({"report_id": rep, "file": f.name, "status": f"unreadable: {type(ex).__name__}: {str(ex)[:60]}"})
                continue
            for sh, raw in sheets.items():
                recs = read_sheet(rep, f, str(sh), raw, log)
                recs_rep += recs
                t = read_coords_sheet(raw)
                if t is not None and len(t):
                    trench_lists.append(t)
        for t in trench_lists:
            recs_rep += link_trenches(recs_rep, t, log, rep)
        recs_rep += apply_gis(rep, rep_dir, recs_rep, log, gis_log, notes)
        # interval rows without coordinates stay in the table output but are not points
        samples += recs_rep
        if not files:
            log.append({"report_id": rep, "status": "no tables in folder (PDF only?)"})
    return samples, log, gis_log


# ----------------------------------------------------------------------------- GIS layers
BLOCK_LON, BLOCK_LAT = (75.5, 78.3), (13.2, 16.3)        # the study block with a margin
ID_FIELD_SKIP = re.compile(r"^(objectid|fid|oid|id|ogc_fid|shape_.*|globalid|ruleid|override)$", re.I)
GIS_AU = re.compile(r"^(au|gold)[\s_]*(ppb|ppm|gpt|g[\s_/]*t|value|assay)?[\s_]*\d?$", re.I)
GIS_SAMPLE = re.compile(r"samp|^s[\s_.]*no|borehole|bh[\s_]*no|trench|^name$|^site", re.I)


def norm_id(s) -> str:
    """Upper case, separators unified: 'hnt 1/1' -> 'HNT-1-1' (keeps 'HNT-1-1' apart from 'HNT-11')."""
    return re.sub(r"[\s\-_./]+", "-", str(s).strip().upper()).strip("-")


def compact(k: str) -> str:
    return k.replace("-", "")


def id_keys(sample: str) -> list[str]:
    """Full sample id, then its trench / borehole part ('HNT-1/3' -> 'HNT-1'; 'GRS-08 4' -> 'GRS-08'),
    then that part without separators ('HNT1') for GIS layers that write ids that way. A part without
    digits ('MS' in 'MS/GR-12') is an area code, not a trench, and is not used."""
    s = str(sample).strip()
    keys = [norm_id(s)]
    segs = [x for x in s.split("/") if x.strip()]
    if len(segs) >= 3:   # 'MS/57F5/OW-8/l': the working 'MS/57F5/OW-8', then 'OW-8' on its own
        keys.append(norm_id("/".join(segs[:-1])))
        mid = norm_id(segs[-2])
        if len(mid) >= 3 and re.search(r"[A-Z]", mid):
            keys += [mid, compact(mid)]
    for sep in ("/", " "):
        if sep in s:
            part = norm_id(s.split(sep)[0])
            keys += [part, compact(part)]
    return [k for k in dict.fromkeys(keys) if len(k) >= 2 and re.search(r"\d", k)]


VW_TAIL = re.compile(r"\s+<?\s*\d*\.?\d+\s*/\s*\d*\.?\d+\s*m\.?\s*$", re.I)        # 'PH/PT-37 0.79/2.5 M'
VW = re.compile(r"(?:^|\s)(<\s*)?(\d*\.\d+|\d+)\s*/\s*(\d*\.?\d+)\s*m\.?\s*$", re.I)    # '0.38/1.5m' = g/t over width
GOLD_TXT = re.compile(r"(?:gold|\bau)\b[^0-9<]{0,25}(<\s*)?(\d*\.?\d+)\s*(g\s*/\s*t|gpt|gm/t|ppm|ppb)", re.I)
TXT_GOLD = re.compile(r"(<\s*)?(\d*\.?\d+)\s*(g\s*/\s*t|gpt|gm/t|ppm|ppb)\s*(?:of\s*)?(?:gold|au)\b", re.I)
TEXT_FIELDS = re.compile(r"remark|descr|^name$|note|comment|result|value", re.I)
COMMODITY_F = re.compile(r"^commodity", re.I)
TYPE_FIELDS = re.compile(r"sample_?type|sample_?media|^type$", re.I)
SAMPLE_LAYERS = re.compile(r"chemical|sample|mineral|old_?work|borehole|bore_?hole|trench|pit|channel|drill", re.I)


def clean_gis_id(v) -> str:
    """GIS sample numbers sometimes carry the result: 'PH/PT-37 0.79/2.5 M' -> 'PH/PT-37'."""
    return VW_TAIL.sub("", str(v)).strip()


def last_key(sample: str) -> str:
    """'PH/BRS-19' -> 'BRS-19' (GIS layers often drop the area prefix). Used only when unique in the report."""
    segs = [x for x in str(sample).split("/") if x.strip()]
    if len(segs) < 2:
        return ""
    k = norm_id(segs[-1])
    return k if len(k) >= 3 and re.search(r"\d", k) and re.search(r"[A-Z]", k) else ""


def text_gold(texts: list[str], vw_ok: bool) -> tuple[float | None, bool, str]:
    """Gold in free text: 'Gold value 3.4 g/t', '120 ppb Au', or (vw_ok) '0.38/1.5m' read as g/t over width -
    only when that is the whole remark or the end of the sample number, so 'pit 3/4 m deep' is not read.
    Returns (ppb, below_detection, the matched text)."""
    for t in texts:
        for rx in (GOLD_TXT, TXT_GOLD):
            m = rx.search(t)
            if m:
                fac = unit_factor(m.group(3))
                if fac:
                    return (0.0 if m.group(1) else float(m.group(2)) * fac), bool(m.group(1)), m.group(0).strip()
    if vw_ok:
        for t in texts:
            m = VW.search(t.strip())
            if m and (m.start() == 0 or re.fullmatch(r"[A-Za-z0-9/\-_ .]+", t[:m.start()].strip() or "x")
                      and not re.search(r"\s", t[:m.start()].strip())):
                return (0.0 if m.group(1) else float(m.group(2)) * 1000.0), bool(m.group(1)), m.group(0).strip()
    return None, False, ""


def guess_crs(g) -> str | None:
    """CRS for a layer without a .prj: decimal degrees or UTM 43N, judged from its extent."""
    b = g.total_bounds
    if not np.all(np.isfinite(b)):
        return None
    if 70 <= b[0] and b[2] <= 85 and 5 <= b[1] and b[3] <= 25:
        return "EPSG:4326"
    if 1e5 <= b[0] and b[2] <= 9e5 and 1.3e6 <= b[1] and b[3] <= 1.9e6:
        return "EPSG:32643"
    return None


def table_points(df: pd.DataFrame) -> gpd.GeoDataFrame | None:
    """A GIS table without geometry but with lon/lat (or UTM easting/northing) columns -> points."""
    num = {c: pd.to_numeric(df[c], errors="coerce") for c in df.columns if c != "geometry"}
    lon = next((c for c in num if LON_LABEL.search(str(c)) and num[c].between(*BLOCK_LON).mean() > 0.8), None)
    lat = next((c for c in num if LAT_LABEL.search(str(c)) and num[c].between(*BLOCK_LAT).mean() > 0.8), None)
    if lon and lat:
        return gpd.GeoDataFrame(df.drop(columns="geometry", errors="ignore"),
                                geometry=gpd.points_from_xy(num[lon], num[lat]), crs="EPSG:4326")
    ex = next((c for c in num if re.search(r"east|^x$|^x_|utm_?x", str(c), re.I) and num[c].between(1e5, 9e5).mean() > 0.8), None)
    no = next((c for c in num if re.search(r"north|^y$|^y_|utm_?y", str(c), re.I) and num[c].between(1.3e6, 1.9e6).mean() > 0.8), None)
    if ex and no:
        return gpd.GeoDataFrame(df.drop(columns="geometry", errors="ignore"),
                                geometry=gpd.points_from_xy(num[ex], num[no]), crs="EPSG:32643")
    return None


def gis_layers(rep_dir: Path) -> tuple[list[tuple[str, gpd.GeoDataFrame, str]], list[str]]:
    """Every readable layer under the report folder: (name, layer, note). Also returns problems."""
    out, problems = [], []
    gdbs = [d for d in rep_dir.rglob("*.gdb") if d.is_dir() and any(c.suffix.lower() in (".gdbtable", ".gdbtablx")
                                                                     for c in d.iterdir())]
    srcs = gdbs + list(rep_dir.rglob("*.shp")) + list(rep_dir.rglob("*.gpkg")) + list(rep_dir.rglob("*.kml"))
    for src in srcs:
        try:
            try:
                import pyogrio
                names = [str(x[0]) for x in pyogrio.list_layers(src)]
            except ImportError:
                import fiona
                names = list(fiona.listlayers(src))
        except Exception as ex:
            problems.append(f"{src.name}: unreadable ({type(ex).__name__}: {str(ex)[:60]})")
            continue
        for nm in names:
            try:
                g = gpd.read_file(src, layer=nm)
            except Exception as ex:
                problems.append(f"{src.name}:{nm}: unreadable ({type(ex).__name__})")
                continue
            if g is None or len(g) == 0:
                continue
            note = ""
            has_geom = isinstance(g, gpd.GeoDataFrame) and "geometry" in g and g.geometry.notna().any()
            if not has_geom:
                g = table_points(pd.DataFrame(g))
                if g is None:
                    continue                      # a plain attribute table (codes, metadata)
                note = "table with coordinate columns"
            g = g[g.geometry.apply(lambda x: x is not None and not x.is_empty)]
            if not len(g):
                continue
            if g.crs is None:
                c = guess_crs(g)
                if c is None:
                    problems.append(f"{src.name}:{nm}: no projection file and the extent is neither degrees nor UTM 43N; skipped")
                    continue
                g = g.set_crs(c)
                note = (note + "; " if note else "") + f"no projection file, assumed {c} from the extent"
            out.append((f"{src.name}:{nm}" if src.suffix.lower() != ".shp" else src.name, g, note))
    return out, problems


def gis_points(g: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Lon / lat of each feature (lines and polygons: a point on the feature); NaN outside the block."""
    gm = g.to_crs("EPSG:32643").geometry
    rep = gm.where(gm.geom_type.isin(["Point", "MultiPoint"]), gm.representative_point())
    ll = gpd.GeoSeries(rep, crs="EPSG:32643").to_crs("EPSG:4326")
    lon, lat = ll.x.values.astype(float), ll.y.values.astype(float)
    bad = ~((lon >= BLOCK_LON[0]) & (lon <= BLOCK_LON[1]) & (lat >= BLOCK_LAT[0]) & (lat <= BLOCK_LAT[1]))
    lon[bad], lat[bad] = np.nan, np.nan
    return lon, lat


def guess_unit(values: list) -> tuple[float | None, str]:
    """Gold field without a unit in its name: g/t if typical values are fractions, ppb if whole numbers >= 5."""
    v = np.array([x for x in values if x is not None and x > 0], dtype=float)
    if len(v) < 3:
        return None, "unit unknown (too few values)"
    med, whole = float(np.median(v)), float(np.mean(np.abs(v - np.round(v)) < 1e-9))
    if med < 1 and whole < 0.5:
        return 1000.0, f"unit guessed g/t (typical value {med:g})"
    if med >= 5 and whole > 0.8:
        return 1.0, f"unit guessed ppb (whole numbers, typical value {med:g})"
    return None, f"unit unclear (typical value {med:g}); not used"


GIS_DUMP: list[pd.DataFrame] = []   # attribute tables of sample-type layers -> gsi_gis_samples.csv


def apply_gis(rep: str, rep_dir: Path, recs: list[dict], log: list, gis_log: list, notes: list[str]) -> list[dict]:
    layers, problems = gis_layers(rep_dir)
    need = [r for r in recs if r["lat"] is None]
    for n in notes + problems:
        gis_log.append({"report_id": rep, "layer": "", "status": n})
    if not layers:
        if not notes and not problems:
            gis_log.append({"report_id": rep, "layer": "", "status": "no GIS files in the folder"})
        else:
            exts = pd.Series([p.suffix.lower() or "(none)" for p in rep_dir.rglob("*") if p.is_file()]).value_counts()
            gis_log.append({"report_id": rep, "layer": "", "status": "no .gdb / .shp layers found; files present: "
                            + ", ".join(f"{k} x{v}" for k, v in exts.head(8).items())})
        return []
    lk_count = pd.Series([last_key(r["sample_no"]) for r in recs]).value_counts()

    def keys_for(r):   # id keys, plus the last segment when no other sample in the report shares it
        lk = last_key(r["sample_no"])
        return id_keys(r["sample_no"]) + ([lk] if lk and lk_count.get(lk, 0) == 1 else [])
    want = {}
    for i, r in enumerate(need):
        for k in keys_for(r):
            want.setdefault(k, []).append(i)
    numeric_keys = {k for k in want if k.isdigit()}
    extra, matched = [], 0
    for name, g, note in layers:
        lon, lat = gis_points(g)
        if SAMPLE_LAYERS.search(name.split(":")[-1]):
            d = pd.DataFrame(g.drop(columns="geometry")).astype(str).head(5000)
            d.insert(0, "lat", np.round(lat[:len(d)], 6))
            d.insert(0, "lon", np.round(lon[:len(d)], 6))
            d.insert(0, "layer", name)
            d.insert(0, "report_id", rep)
            GIS_DUMP.append(d)
        row = {"report_id": rep, "layer": name, "geometry": ",".join(sorted(g.geom_type.dropna().unique())),
               "features": len(g), "crs": str(g.crs.to_epsg() or g.crs.name) if g.crs else "", "note": note,
               "outside_block": int(np.isnan(lon).sum()), "fields": ",".join(str(c) for c in g.columns if c != "geometry")[:300]}
        # (a) an id field whose values match the table's sample / trench / borehole numbers
        best_f, best_hits = None, 0
        for f in g.columns:
            if f == "geometry" or g[f].dtype.kind in "fcb" or (ID_FIELD_SKIP.match(str(f)) and (
                    g[f].dtype.kind in "iu" or g[f].astype(str).str.fullmatch(r"\d+").all())):
                continue                          # running feature numbers, not sample ids
            vals = g[f].astype(str).map(clean_gis_id).map(norm_id)
            cand = set(vals) | set(vals.map(compact))
            ok_keys = set(want) if GIS_SAMPLE.search(str(f)) else set(want) - numeric_keys   # bare numbers only in a sample field
            hits = len(cand & ok_keys)
            if hits > best_hits:
                best_f, best_hits = f, hits
        n0 = matched
        if best_f is not None and best_hits >= 2:
            pos = {}
            for k, x, y in zip(g[best_f].astype(str).map(clean_gis_id).map(norm_id), lon, lat):
                if x == x:
                    pos.setdefault(k, (x, y))
                    pos.setdefault(compact(k), (x, y))
            for r in need:
                if r["lat"] is not None:
                    continue
                for k in keys_for(r):
                    if k.isdigit() and not GIS_SAMPLE.search(str(best_f)):
                        continue
                    if k in pos:
                        r["lon"], r["lat"] = map(float, pos[k])
                        r["source"] = f"coordinates from GIS {name} ({best_f})"
                        matched += 1
                        break
            log.append({"report_id": rep, "file": name, "sheet": "", "status":
                        f"GIS layer gave coordinates to {matched - n0} table samples (matched on {best_f})"})
        row.update({"id_field": best_f if best_hits >= 2 else "", "ids_matched": best_hits if best_hits >= 2 else 0,
                    "table_samples_located": matched - n0})
        # (b) a gold field in the layer itself
        au_f = next((f for f in g.columns if f != "geometry" and not NOT_AU.search(str(f))
                     and (AU_LABEL.search(str(f).replace("_", " ")) or GIS_AU.match(str(f)))
                     and g[f].dtype.kind in "fciuO"), None)
        n_add, unit_note = 0, ""
        if au_f:
            parsed = [parse_value(x) for x in g[au_f]]
            fac = unit_factor(str(au_f))
            unit_note = "unit from field name" if fac else ""
            if not fac:
                fac, unit_note = guess_unit([v for v, b in parsed if not b])
            if fac:
                idf = next((f for f in g.columns if f != "geometry" and (SAMPLE_LABEL.search(str(f).replace("_", " "))
                                                                        or GIS_SAMPLE.search(str(f)))), None)
                located = {norm_id(r["sample_no"]) for r in recs if r["lat"] is not None}
                for j, (x, y) in enumerate(zip(lon, lat)):
                    v, bdl = parsed[j]
                    if v is None or x != x or (idf and norm_id(clean_gis_id(g[idf].iloc[j])) in located):   # already located
                        continue
                    sid = clean_gis_id(g[idf].iloc[j]) if idf else ""
                    extra.append({"report_id": rep, "file": name, "sheet": "GIS", "sample_no": sid,
                                  "sample_type": (lambda t: t if t != "unknown" else type_from_layer(name))(type_from(name, sid)),
                                  "au_raw": str(g[au_f].iloc[j]),
                                  "au_ppb": v * fac, "below_detection": bdl, "lon": float(x), "lat": float(y),
                                  "from_m": None, "to_m": None, "source": f"GIS {name} ({au_f})"})
                    n_add += 1
                log.append({"report_id": rep, "file": name, "sheet": "", "status": f"GIS layer with gold field {au_f}: {n_add} samples"})
        # (c) gold written into text: REMARKS 'Gold value 3.4 g/t', '0.38/1.5m', or a result after the sample number
        if not n_add and SAMPLE_LAYERS.search(name.split(":")[-1]):
            tf = [f for f in g.columns if f != "geometry" and TEXT_FIELDS.search(str(f)) and g[f].dtype.kind == "O"]
            idf = next((f for f in g.columns if f != "geometry" and re.search(r"sample_?n(o|um)", str(f), re.I)), None)
            cf = next((f for f in g.columns if COMMODITY_F.match(str(f))), None)
            ty = [f for f in g.columns if TYPE_FIELDS.search(str(f))]
            located = {norm_id(r["sample_no"]) for r in recs if r["lat"] is not None}
            lay = name.split(":")[-1]
            n_txt = 0
            for j, (x, y) in enumerate(zip(lon, lat)):
                if x != x:
                    continue
                comm = str(g[cf].iloc[j]) if cf else ""
                if has(comm) and not re.search(r"gold|\bau\b", comm, re.I):
                    continue                                   # another commodity's result
                texts = [str(g[f].iloc[j]) for f in tf if has(g[f].iloc[j])]
                sid_raw = str(g[idf].iloc[j]) if idf and has(g[idf].iloc[j]) else ""
                v, bdl, hit = text_gold(texts + ([sid_raw] if sid_raw else []), vw_ok=True)
                if v is None:
                    continue
                sid = clean_gis_id(sid_raw)
                if sid and norm_id(sid) in located:
                    continue
                ttxt = " ".join(str(g[f].iloc[j]) for f in ty if has(g[f].iloc[j]))
                st = type_from(ttxt, sid) if (ttxt or sid) else "unknown"
                if st == "unknown":
                    st = type_from_layer(name)
                if st == "unknown" and (re.search(r"grab|chip|rock|lode|vein|reef", ttxt, re.I)
                                        or re.search(r"mineral|old_?work", lay, re.I)):
                    st = "rock_chip"                           # grab / rock media, or a result at a mineralised spot
                extra.append({"report_id": rep, "file": name, "sheet": "GIS text", "sample_no": sid or f"{lay}#{j + 1}",
                              "sample_type": st, "au_raw": hit, "au_ppb": v, "below_detection": bdl,
                              "lon": float(x), "lat": float(y), "from_m": None, "to_m": None,
                              "source": f"GIS {name} (gold read from text)"})
                n_txt += 1
            if n_txt:
                n_add = n_txt
                au_f, unit_note = "(text)", "read from remarks / sample number; value/width taken as g/t"
                log.append({"report_id": rep, "file": name, "sheet": "", "status": f"GIS layer text gave {n_txt} gold results"})
        row.update({"gold_field": au_f or "", "gold_unit": unit_note, "gis_samples_added": n_add})
        if not row["id_field"] and not n_add:
            # show what the layer's most id-like field holds next to what the tables need, to fix by hand
            best_txt, best_n = "", 0
            for f in g.columns:
                if f == "geometry":
                    continue
                v = g[f].astype(str)
                v = v[v.str.contains(r"\d") & v.str.contains(r"[A-Za-z]")]
                if v.nunique() > best_n:
                    best_txt, best_n = f"{f}: " + ", ".join(v.drop_duplicates().head(4)), v.nunique()
            row["gis_ids_example"] = best_txt[:120]
            row["table_ids_example"] = ", ".join(list(dict.fromkeys(str(r["sample_no"]) for r in need))[:6])[:120]
        gis_log.append(row)
    return extra


# ----------------------------------------------------------------------------- sites
def make_sites(pts: pd.DataFrame) -> pd.DataFrame:
    """Merge points within SITE_M (single linkage on a grid) and label each site."""
    xy = np.array([TO_UTM.transform(lo, la) for lo, la in zip(pts["lon"], pts["lat"])])
    n = len(pts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    cell = {}
    for i, (x, y) in enumerate(xy):
        cell.setdefault((int(x // SITE_M), int(y // SITE_M)), []).append(i)
    for (cx, cy), idx in cell.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((cx + dx, cy + dy), []):
                    for i in idx:
                        if i < j and (xy[i, 0] - xy[j, 0]) ** 2 + (xy[i, 1] - xy[j, 1]) ** 2 <= SITE_M ** 2:
                            parent[find(i)] = find(j)
    pts = pts.assign(site=[find(i) for i in range(n)], x=xy[:, 0], y=xy[:, 1])
    rows = []
    for sid, g in pts.groupby("site"):
        rock = g[g["sample_type"].isin(ROCK)]
        strong = g[g["sample_type"].isin(STRONG)]
        best_rock = rock["au_ppb"].max() if len(rock) else np.nan
        best_all = g["au_ppb"].max()
        if len(rock) and best_rock >= POS_PPB:
            lab = "positive"
        elif (len(rock) and best_rock >= ANOM_PPB) or best_all >= ANOM_PPB:
            lab = "anomalous"
        elif len(strong) >= 3 and strong["au_ppb"].max() < BARREN_PPB and best_all < BARREN_PPB:
            lab = "barren_strong"
        elif len(rock) >= 3 and best_rock < BARREN_PPB and best_all < BARREN_PPB:
            lab = "barren"
        elif len(rock) == 0 and len(g) >= 1 and best_all < WEAK_PPB:
            lab = "barren_weak"
        else:
            lab = "too_few_samples"
        top = g.loc[g["au_ppb"].idxmax()]
        w = g if lab != "positive" else g[g["au_ppb"] >= POS_PPB]
        rows.append({"label": lab, "best_au_ppb": best_all, "best_sample": top["sample_no"], "best_type": top["sample_type"],
                     "n_samples": len(g), "n_rock": len(rock), "n_trench_pit_drill": len(strong),
                     "types": ",".join(sorted(g["sample_type"].unique())), "reports": ",".join(sorted(g["report_id"].unique())),
                     "x": float(w["x"].mean()), "y": float(w["y"].mean())})
    s = pd.DataFrame(rows)
    to_ll = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
    ll = np.array([to_ll.transform(x, y) for x, y in zip(s["x"], s["y"])])
    s["lon"], s["lat"] = ll[:, 0], ll[:, 1]
    order = {"positive": 0, "anomalous": 1, "barren_strong": 2, "barren": 3, "barren_weak": 4, "too_few_samples": 5}
    return s.assign(_o=s["label"].map(order)).sort_values(["_o", "best_au_ppb"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)


def has(v) -> bool:
    return v is not None and str(v).strip() not in ("", "nan", "None")


def km(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2) / 1000.0


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="GSI report tables -> gold samples, label sites, prospect evidence.")
    ap.add_argument("processed")
    ap.add_argument("--reports-dir", default="Reports")
    ap.add_argument("--model", default="pg_ens")
    args = ap.parse_args()
    proc, root = Path(args.processed), Path(args.reports_dir)
    if not root.exists():
        sys.exit(f"Reports folder not found: {root}")

    samples, log, gis_log = read_reports(root)
    man = proc / "gsi_manual_points.csv"
    if man.exists():
        m = pd.read_csv(man)
        for _, r in m.iterrows():
            samples.append({"report_id": r["report_id"], "file": "gsi_manual_points.csv", "sheet": r.get("page_ref", ""),
                            "sample_no": str(r["sample_no"]), "sample_type": r["sample_type"], "au_raw": str(r.get("au_raw", "")),
                            "au_ppb": float(r["au_ppb"]), "below_detection": bool(r.get("below_detection", False)),
                            "lon": float(r["lon"]), "lat": float(r["lat"]), "from_m": None, "to_m": None,
                            "source": str(r.get("source", "typed from PDF"))})
        log.append({"report_id": ",".join(sorted(m["report_id"].unique())), "file": man.name, "status": f"{len(m)} points typed from PDFs"})
    if not samples:
        sys.exit("No gold samples found.")
    samples, n_dup = dedupe_gis(samples)
    if n_dup:
        log.append({"report_id": "", "file": "", "sheet": "", "status": f"{n_dup} duplicate GIS samples dropped "
                    "(same shapefile in several folders, or review copies)"})
    df = pd.DataFrame(samples)
    df.to_csv(proc / "gsi_samples.csv", index=False, encoding="utf-8-sig")
    # per-report summary: what each report found, with or without coordinates
    summ = []
    for rid, g in df.groupby("report_id"):
        gk = g[g["au_ppb"].notna()]
        named = gk[gk["sample_no"].astype(str).str.strip().replace("nan", "") != ""]   # not summary-table rows
        src = named if len(named) else gk
        top = src.loc[src["au_ppb"].idxmax()] if len(src) else None
        hi = gk[gk["au_ppb"] >= POS_PPB]
        summ.append({"report_id": rid, "gold_results": len(g), "with_unit": len(gk),
                     "with_coordinates": int((g["lat"].notna() & g["au_ppb"].notna()).sum()),
                     "n_ge_0_5_located": int(hi["lat"].notna().sum()),
                     "best_g_t": round(top["au_ppb"] / 1000, 3) if top is not None else None,
                     "best_sample": "" if top is None else f"{top['sample_no']} ({top['sample_type']}, {top['sheet']})",
                     "n_ge_0_5_g_t": int((gk["au_ppb"] >= POS_PPB).sum()), "n_ge_0_1_g_t": int((gk["au_ppb"] >= ANOM_PPB).sum()),
                     "types": ",".join(sorted(g["sample_type"].unique()))})
    summ = pd.DataFrame(summ)
    summ.to_csv(proc / "gsi_report_summary.csv", index=False, encoding="utf-8-sig")
    # gold >= 0.5 g/t still without a position: what the GIS files (or a PDF map) should place
    unl = df[(df["au_ppb"] >= POS_PPB) & df["lat"].isna()].sort_values(["report_id", "au_ppb"], ascending=[True, False])
    unl[["report_id", "sample_no", "sample_type", "au_ppb", "au_raw", "sheet", "file"]].to_csv(
        proc / "gsi_unlocated_gold.csv", index=False, encoding="utf-8-sig")
    gl = pd.DataFrame(gis_log)
    gl.to_csv(proc / "gsi_gis_layers.csv", index=False, encoding="utf-8-sig")
    if GIS_DUMP:   # every feature of the sample-type layers, all attributes: to fix matching from real values
        pd.concat(GIS_DUMP, ignore_index=True).to_csv(proc / "gsi_gis_samples.csv", index=False, encoding="utf-8-sig")
    pts = df[df["lat"].notna() & df["lon"].notna() & df["au_ppb"].notna()].copy()
    pts = pts[pts["lat"].between(5, 40) & pts["lon"].between(60, 100)]
    gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326").to_file(
        proc / "gsi_samples.gpkg", driver="GPKG")

    sites = make_sites(pts.reset_index(drop=True))
    # distance to the existing gold labels
    try:
        lab = gpd.read_file(proc / "ka_ap_v1.gpkg", layer="gold_labels").to_crs("EPSG:32643")
        lx, ly = lab.geometry.x.values, lab.geometry.y.values
        sites["dist_known_gold_km"] = [float(np.min(np.hypot(lx - x, ly - y))) / 1000 for x, y in zip(sites["x"], sites["y"])]
    except Exception:
        sites["dist_known_gold_km"] = np.nan
    sites["new_label"] = (sites["label"] == "positive") & (sites["dist_known_gold_km"] > 1.0)
    sites.to_csv(proc / "gsi_label_sites.csv", index=False, encoding="utf-8-sig")
    gpd.GeoDataFrame(sites, geometry=gpd.points_from_xy(sites["lon"], sites["lat"]), crs="EPSG:4326").to_file(
        proc / "gsi_label_sites.gpkg", driver="GPKG")

    # per-prospect evidence
    mdir = proc / "models" / args.model
    pr_p = mdir / "step4" / "prospects_after_check.csv"
    pr = pd.read_csv(pr_p) if pr_p.exists() else pd.read_csv(mdir / "prospects.csv")
    tcol = "tier_after" if "tier_after" in pr else "best_tier"
    pr = pr[pr[tcol].isin(["A", "B"])]
    ev = []
    pxy = np.array([TO_UTM.transform(lo, la) for lo, la in zip(pts["lon"], pts["lat"])])
    for _, p in pr.iterrows():
        px, py = TO_UTM.transform(p["peak_lon"], p["peak_lat"])
        d_s = np.hypot(pxy[:, 0] - px, pxy[:, 1] - py) / 1000
        near1 = pts[d_s <= 1.0]
        d_site = np.hypot(sites["x"] - px, sites["y"] - py) / 1000
        pos = sites[sites["label"] == "positive"]
        bar = sites[sites["label"].str.startswith("barren")]
        dp = d_site[pos.index].min() if len(pos) else np.nan
        bp = pos.loc[d_site[pos.index].idxmin()] if len(pos) else None
        nb2 = int((d_site[bar.index] <= 2.0).sum())
        nbs2 = int((d_site[bar[bar["label"] == "barren_strong"].index] <= 2.0).sum())
        if len(near1) and near1["au_ppb"].max() >= POS_PPB:
            verdict = f"gold confirmed within 1 km of the peak ({near1['au_ppb'].max() / 1000:.2f} g/t)"
        elif len(near1) >= 3 and near1["au_ppb"].max() < BARREN_PPB:
            verdict = f"tested within 1 km of the peak and barren ({len(near1)} samples, best {near1['au_ppb'].max():.0f} ppb)"
        elif dp == dp and dp <= 3.0:
            verdict = f"not tested at the peak; gold found {dp:.1f} km away ({bp['best_au_ppb'] / 1000:.2f} g/t, {bp['reports']})"
        elif nb2:
            verdict = f"not tested at the peak; {nb2} barren sites within 2 km"
        else:
            verdict = "untested by the reports read so far"
        ev.append({"prospect": p["prospect"], "tier": p[tcol], "peak_lon": p["peak_lon"], "peak_lat": p["peak_lat"],
                   "dist_known_gold_km": p.get("dist_known_gold_km", np.nan),
                   "samples_within_1km": len(near1), "best_within_1km_ppb": near1["au_ppb"].max() if len(near1) else np.nan,
                   "nearest_positive_km": round(dp, 2) if dp == dp else np.nan,
                   "nearest_positive": "" if bp is None else f"{bp['best_sample']} {bp['best_au_ppb'] / 1000:.2f} g/t ({bp['reports']})",
                   "barren_sites_within_2km": nb2, "barren_strong_within_2km": nbs2, "verdict": verdict})
    evd = pd.DataFrame(ev)
    # block-level evidence: which downloaded reports' exploration blocks contain each prospect peak, and
    # what they found (works even for tables without sample coordinates)
    try:
        blocks = gpd.read_file(proc / "ka_ap_v1.gpkg", layer="exploration_blocks").to_crs("EPSG:32643")
        have = set(summ["report_id"]) | {d.name for d in root.iterdir() if d.is_dir()}
        blocks = blocks[blocks["report_id"].isin(have)]
        best = dict(zip(summ["report_id"], summ["best_g_t"]))
        fnd_best = {}
        for f in sorted(proc.glob("gsi_findings_*.csv")):
            for _, r in pd.read_csv(f, dtype=str, keep_default_na=False).iterrows():
                fnd_best[r["report_id"]] = r.get("best_result", "")
        pgeom = {}
        if (mdir / "prospects.gpkg").exists():   # prospect outlines, if the model folder has them
            pg_df = gpd.read_file(mdir / "prospects.gpkg").to_crs("EPSG:32643")
            pgeom = dict(zip(pg_df["prospect"], pg_df.geometry))
        cover = []
        for _, p in pr.iterrows():
            pt = gpd.points_from_xy([p["peak_lon"]], [p["peak_lat"]], crs="EPSG:4326").to_crs("EPSG:32643")[0]
            pg = pgeom.get(p["prospect"])
            hit = blocks[blocks.geometry.intersects(pg) if pg is not None else blocks.geometry.buffer(500).contains(pt)]
            hit = hit.sort_values("area_km2").drop_duplicates("report_id")
            parts = []
            for _, b in hit.iterrows():
                g = best.get(b["report_id"])
                txt = f"best {g:g} g/t" if g is not None and g == g else (fnd_best.get(b["report_id"], "")[:40] or "no gold table")
                parts.append(f"{b['report_id']} ({b.get('stage', '')}, {b['area_km2']:.0f} km2: {txt})")
            cover.append("; ".join(parts))
        evd["reports_covering_peak"] = cover
    except Exception as ex:
        print(f"  (block-level evidence skipped: {type(ex).__name__})")
    evd.to_csv(proc / "gsi_prospect_evidence.csv", index=False, encoding="utf-8-sig")

    # merge typed-up report findings into report_findings.csv (only blank rows are filled)
    rf_p = proc / "report_findings.csv"
    fnd = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in sorted(proc.glob("gsi_findings_*.csv"))]
    if fnd:
        fnd = pd.concat(fnd).drop_duplicates("report_id", keep="last")
        rf = pd.read_csv(rf_p, dtype=str, keep_default_na=False) if rf_p.exists() else pd.DataFrame(columns=["report_id"])
        for col in fnd.columns:
            if col not in rf.columns:
                rf[col] = ""
        filled = 0
        for _, r in fnd.iterrows():
            hit = rf.index[rf["report_id"] == r["report_id"]]
            if len(hit) == 0:
                rf = pd.concat([rf, pd.DataFrame([r])], ignore_index=True)
                filled += 1
                continue
            i = hit[0]
            if str(rf.at[i, "outcome"]).strip() and not str(rf.at[i, "reviewed"]).startswith("yes (Claude"):
                continue   # the user filled it in themselves: never overwrite
            for col in fnd.columns:
                if r[col] != "":
                    rf.at[i, col] = r[col]
            filled += 1
        rf.to_csv(rf_p, index=False, encoding="utf-8-sig")
        print(f"report_findings.csv: {filled} reports filled in from {', '.join(f.name for f in sorted(proc.glob('gsi_findings_*.csv')))}")

    # summary
    lg = pd.DataFrame(log)
    print("\nTables read:")
    for _, r in lg.iterrows():
        extra = ""
        if r.get("status", "").startswith("read"):
            extra = f"  {int(r.get('samples', 0))} samples, {int(r.get('with_coords', 0))} with coordinates, unit {r.get('unit', '?')}"
        print(f"  {r.get('report_id', ''):16} {str(r.get('sheet', ''))[:30]:30} {r.get('status', '')}{extra}")
    print("\nPer report (best gold result; points need coordinates):")
    for _, r in summ.iterrows():
        print(f"  {r['report_id']:16} {int(r['gold_results']):5} results, {int(r['with_coordinates']):4} with coordinates, "
              f"best {r['best_g_t'] if r['best_g_t'] is not None else '-'} g/t, {int(r['n_ge_0_5_g_t'])} >= 0.5 g/t "
              f"({int(r['n_ge_0_5_located'])} located)  [{r['best_sample'][:50]}]")
    print("\nGIS files per report (full detail in gsi_gis_layers.csv):")
    for rid, g in gl.groupby("report_id", sort=True):
        lay = g[g["layer"].astype(str) != ""] if "layer" in g else g.iloc[0:0]
        msgs = g[g["layer"].astype(str) == ""]["status"].dropna().tolist() if "status" in g else []
        if len(lay):
            n_tab = int(lay["table_samples_located"].fillna(0).sum())
            n_gis = int(lay["gis_samples_added"].fillna(0).sum())
            print(f"  {rid:16} {len(lay)} layers read: {n_tab} table samples located, {n_gis} samples from GIS gold fields")
            for _, r in lay.iterrows():
                bits = []
                if has(r.get("id_field")):
                    bits.append(f"ids on {r['id_field']} -> {int(r['table_samples_located'])} located")
                if has(r.get("gold_field")):
                    bits.append(f"gold field {r['gold_field']} ({r.get('gold_unit', '')}) -> {int(r['gis_samples_added'])} samples")
                if not bits and has(r.get("gis_ids_example")):
                    bits.append(f"no match; layer ids look like [{r['gis_ids_example'][:70]}]")
                if has(r.get("note")):
                    bits.append(str(r["note"]))
                if bits:
                    print(f"      {str(r['layer'])[:45]:45} {int(r['features']):5} {str(r['geometry'])[:12]:12} " + "; ".join(bits))
            need = summ.loc[summ["report_id"] == rid, "with_unit"].sum() - summ.loc[summ["report_id"] == rid, "with_coordinates"].sum()
            if need > 0 and n_tab == 0:
                ex = lay["table_ids_example"].dropna()
                ex = ex[ex != ""]
                if len(ex):
                    print(f"      table samples still without position look like [{ex.iloc[0][:80]}]")
        for m_ in msgs:   # unpack notes matter only when something failed or no layers came out
            if not len(lay) or "unpacked (" not in m_:
                print(f"  {rid:16} {m_}")
    if len(unl):
        print(f"\nGold >= 0.5 g/t still without a position: {len(unl)} results "
              f"({', '.join(f'{k} {v}' for k, v in unl['report_id'].value_counts().items())}); listed in gsi_unlocated_gold.csv")
    print(f"\n{len(df)} gold results, {len(pts)} with coordinates -> {len(sites)} sites:")
    print("  " + ", ".join(f"{k} {v}" for k, v in sites["label"].value_counts().items()))
    pos = sites[sites["label"] == "positive"]
    for _, s in pos.iterrows():
        print(f"  positive {s['lon']:.5f} {s['lat']:.5f}  best {s['best_au_ppb'] / 1000:.2f} g/t ({s['best_type']}, {s['reports']})"
              f"  {s['dist_known_gold_km']:.1f} km from the nearest existing gold label{'  -> NEW LABEL' if s['new_label'] else ''}")
    print("\nProspects with evidence:")
    for _, e in evd[evd["verdict"] != "untested by the reports read so far"].iterrows():
        print(f"  {e['prospect']} (tier {e['tier']}): {e['verdict']}")
    print(f"  ({(evd['verdict'] == 'untested by the reports read so far').sum()} of {len(evd)} prospects untested by sample points)")
    if "reports_covering_peak" in evd:
        print("Downloaded reports whose exploration block overlaps a prospect (no sample coordinates needed):")
        for _, e in evd[evd["reports_covering_peak"] != ""].iterrows():
            print(f"  {e['prospect']} (tier {e['tier']}): {e['reports_covering_peak'][:200]}")
    try:   # quicklook map
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8.5), dpi=120)
        style = {"barren_weak": ("#bdbdbd", 10, "o"), "too_few_samples": ("#e0e0e0", 8, "o"), "barren": ("#4575b4", 22, "s"),
                 "barren_strong": ("#313695", 34, "s"), "anomalous": ("#fdae61", 40, "^"), "positive": ("#d73027", 120, "*")}
        for labl, (c, sz, mk) in style.items():
            g = sites[sites["label"] == labl]
            if len(g):
                ax.scatter(g["lon"], g["lat"], s=sz, c=c, marker=mk, label=f"{labl} ({len(g)})", zorder=3, edgecolors="none")
        ax.scatter(pr["peak_lon"], pr["peak_lat"], s=60, facecolors="none", edgecolors="k", linewidths=1.2,
                   label="tier A/B prospect peak", zorder=4)
        for _, p in pr.iterrows():
            ax.annotate(p["prospect"], (p["peak_lon"], p["peak_lat"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        try:
            lab_ll = lab.to_crs("EPSG:4326")
            ax.scatter(lab_ll.geometry.x, lab_ll.geometry.y, s=14, c="#1a9850", marker="x", label="existing gold labels", zorder=2)
        except Exception:
            pass
        ax.set_xlim(75.98, 77.77)
        ax.set_ylim(13.73, 15.77)
        ax.set_aspect(1 / math.cos(math.radians(14.75)))
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.set_title("GSI report evidence: label sites and prospects")
        fig.tight_layout()
        fig.savefig(proc / "gsi_label_sites.png")
        plt.close(fig)
    except Exception as ex:
        print(f"  (map skipped: {type(ex).__name__})")
    print(f"\nWritten in {proc}: gsi_samples.csv/.gpkg, gsi_report_summary.csv, gsi_label_sites.csv/.gpkg/.png, "
          "gsi_prospect_evidence.csv, gsi_gis_layers.csv, gsi_gis_samples.csv, gsi_unlocated_gold.csv")


if __name__ == "__main__":
    main()
