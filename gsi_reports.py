#!/usr/bin/env python3
"""
gsi_reports.py - exactly which GSI report files to fetch from NGDR, and what is inside them once fetched.

    python gsi_reports.py fetchlist data/processed --model pg_ens
    python gsi_reports.py collect data/processed --from "%USERPROFILE%\\Downloads"
    python gsi_reports.py inventory data/processed

fetchlist  writes data/processed/gsi_fetch_list.csv: one row per file to download, in fetch order,
           with its NGDR folder path and where to save it, plus GSI_FETCH_README.txt.
           Which reports: every GSI gold report whose block covers a tier A or B prospect.
           Order (most useful first):
             1. new-ground tier A prospects (>= 5 km from known gold): has anyone tested this ground?
             2. other tier A prospects
             3. tier B only
             4. regional surveys (> 500 km2), which give background context
           Inside a batch: reports with digital tables first, follow-up stages (G3, G2, G1) before
           reconnaissance (G4), smaller (more detailed) blocks first.
           Which files in each report:
             TABLES/*.xlsx|xls|zip    needed   the analytical results (gold assays, sample numbers)
             GIS_FILES/*.gdb.zip|zip  needed   sample, trench and borehole locations, detailed geology
             TEXT/*.pdf               needed   read the summary and conclusions (one PDF; the .docx is a copy)
             GEOREF_IMAGES/*.tif      optional scanned plates; large, only if the GIS files are missing
collect    the easy way to do the downloading: download the files from the NGDR portal into any one
           folder (your Downloads folder is fine), any order, without renaming or sorting them.
           collect finds every listed file there by name (also "name (1).xlsx" copies, and files
           that only carry the report id), copies it to its place under data/raw/gsi_reports/,
           ticks 'downloaded' in gsi_fetch_list.csv and prints what is still missing per report.
inventory  looks inside everything saved under data/raw/gsi_reports/<report_id>/ (unzips the GIS
           files next to them) and writes data/processed/gsi_inventory.csv: every table sheet and GIS
           layer with its rows, columns, coordinate columns and gold-assay columns (with the maximum
           value), so the gold results can be turned into V2 labels without opening each file by hand.
           Reading .xls needs: pip install xlrd   (.xlsx needs openpyxl, usually already installed)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from pathlib import Path

try:
    import geopandas as gpd
    import pandas as pd
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

NEW_GROUND_KM = 5.0
REGIONAL_KM2 = 500.0
KIND = {"TABLES": "tables", "GIS_FILES": "gis", "TEXT": "text", "GEOREF_IMAGES": "plates"}
WHY = {"tables": "analytical results: gold assays with sample numbers",
       "gis": "sample / trench / borehole locations and detailed geology",
       "text": "summary, conclusions, recommendations; check sampling method and cut-offs",
       "plates": "scanned maps; only needed if the GIS files are missing or empty"}
AU_COL = re.compile(r"(^|[\s_(/])(au|gold)([\s_)/]|$)|au\s*\(|au_?pp[bm]|au_?g_?t|g/t|gpt|ppb", re.I)
XY_COL = re.compile(r"^(lat|latitude|long|longitude|lon|easting|northing|east|north|x|y|x_coord|y_coord|"
                    r"utm_?e|utm_?n|xcoord|ycoord)(\b|_)", re.I)


def split_paths(s) -> list[str]:
    s = str(s or "")
    return [] if s in ("", "None", "nan") else [p.strip() for p in s.split(",") if p.strip()]


def stage_rank(stage: str) -> int:
    st = str(stage or "").upper()
    return {"G1": 0, "G2": 1, "G3": 2}.get(st, 3 if st.startswith("G4") else 2 if st.startswith("E") else 4)


# ----------------------------------------------------------------------------- fetch list
def fetchlist(proc: Path, model: str) -> None:
    mdir = proc / "models" / model
    blocks = gpd.read_file(proc / "ka_ap_v1.gpkg", layer="exploration_blocks", ignore_geometry=True)
    blocks = blocks.sort_values("area_km2").drop_duplicates("report_id")   # a report can have several blocks
    pr = pd.read_csv(mdir / "prospects.csv")
    after = mdir / "step4" / "prospects_after_check.csv"
    if after.exists():   # use the desk-checked tiers
        a = pd.read_csv(after)[["prospect", "tier_after"]]
        pr = pr.merge(a, on="prospect", how="left")
        pr["tier"] = pr["tier_after"].fillna(pr["best_tier"])
    else:
        pr["tier"] = pr["best_tier"]
    pr = pr[pr["tier"].isin(["A", "B"])]
    tier = dict(zip(pr["prospect"], pr["tier"]))
    newg = set(pr.loc[(pr["tier"] == "A") & (pr["dist_known_gold_km"] >= NEW_GROUND_KM), "prospect"])
    rp = pd.read_csv(mdir / "reports_to_pull.csv")
    # one row per report (a report can cover several exploration blocks): union of the prospects
    rp = (rp.assign(prospects=rp["prospects"].astype(str))
            .groupby("report_id", as_index=False)["prospects"]
            .agg(lambda s: " ".join(sorted(set(" ".join(s).split())))))

    reps = []
    for _, r in rp.iterrows():
        ps = [p for p in str(r["prospects"]).split() if p in tier]
        if not ps:
            continue
        b = blocks[blocks["report_id"] == r["report_id"]]
        if b.empty:
            continue
        b = b.iloc[0]
        area = float(b["area_km2"]) if pd.notna(b["area_km2"]) else float("nan")
        if area > REGIONAL_KM2:
            batch = 4
        elif any(p in newg for p in ps):
            batch = 1
        elif any(tier[p] == "A" for p in ps):
            batch = 2
        else:
            batch = 3
        files = {k: [] for k in KIND.values()}
        cut = False
        for col in ("report_files", "table_files", "gis_files"):
            for path in split_paths(b[col]):
                if not re.search(r"\.(pdf|docx?|xlsx?|csv|zip|tiff?|jpe?g|png|kmz|gpkg)$", path, re.I):
                    cut = True      # the index field was cut off mid-name (254-character limit)
                    continue
                parts = path.replace("\\", "/").split("/")
                folder = next((KIND[x] for x in parts if x in KIND), None) or ("text" if col == "report_files" else
                                                                                 "tables" if col == "table_files" else "gis")
                files[folder].append(path)
        # one text file is enough: prefer the PDF
        pdfs = [f for f in files["text"] if f.lower().endswith(".pdf")]
        files["text"] = pdfs[:2] if pdfs else files["text"][:1]
        reps.append({"report_id": r["report_id"], "batch": batch, "prospects": " ".join(ps),
                     "new_ground": " ".join(p for p in ps if p in newg), "stage": b.get("stage", ""),
                     "year": b.get("year_start", ""), "area_km2": area, "title": b.get("title", ""),
                     "has_tables": bool(files["tables"]), "has_gis": bool(files["gis"]), "files": files,
                     "more": cut})
    reps.sort(key=lambda x: (x["batch"], not x["has_tables"], stage_rank(x["stage"]), x["area_km2"]))

    rows = []
    for order, rep in enumerate(reps, 1):
        for kind in ("tables", "gis", "text", "plates"):
            for path in rep["files"][kind]:
                name = path.replace("\\", "/").split("/")[-1]
                rel = path.replace("\\", "/").split("Exploration_Data/")[-1]
                rows.append({"fetch_order": order, "batch": rep["batch"], "report_id": rep["report_id"],
                             "prospects": rep["prospects"], "new_ground": rep["new_ground"], "stage": rep["stage"],
                             "year": rep["year"], "area_km2": round(rep["area_km2"], 1), "kind": kind,
                             "needed": "optional" if kind == "plates" else "yes", "file_name": name,
                             "ngdr_path": "Exploration_Data/" + rel,
                             "save_as": f"data/raw/gsi_reports/{rep['report_id']}/{kind}/{name}",
                             "why": WHY[kind], "more_in_folder": "yes" if rep["more"] else "",
                             "title": rep["title"], "downloaded": ""})
    out = proc / "gsi_fetch_list.csv"
    old = {}
    if out.exists():   # keep the user's "downloaded" ticks
        try:
            o = pd.read_csv(out, dtype=str, keep_default_na=False)
            old = dict(zip(o["save_as"], o["downloaded"]))
        except Exception:
            old = {}
    for r in rows:
        r["downloaded"] = old.get(r["save_as"], "")
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["note"])
        w.writeheader()
        w.writerows(rows)

    label = {1: "new-ground tier A", 2: "other tier A", 3: "tier B only", 4: "regional (context)"}
    lines = ["GSI report fetch list (from gsi_reports.py fetchlist). Save each file at its save_as path.", ""]
    for b in (1, 2, 3, 4):
        rs = [r for r in reps if r["batch"] == b]
        if not rs:
            continue
        nf = sum(1 for r in rows if r["batch"] == b and r["needed"] == "yes")
        lines.append(f"Batch {b} - {label[b]}: {len(rs)} reports, {nf} files needed")
        for r in rs:
            kinds = "+".join(k for k in ("tables", "gis", "text") if r["files"][k]) or "text only"
            lines.append(f"  {r['report_id']:16} {str(r['stage']):4} {r['area_km2']:>7.1f} km2  {kinds:16} "
                         f"prospects {r['prospects']}{'  (new ground: ' + r['new_ground'] + ')' if r['new_ground'] else ''}")
        lines.append("")
    lines += ["How to fetch: on geodataindia.gov.in (logged in), search each report_id and download everything",
              "in its TABLES, GIS_FILES and TEXT folders into one folder (Downloads is fine). Do not rename or sort.",
              "(more_in_folder = yes: the index lists only some of that report's files; the portal folder has all.)",
              "Then run: python gsi_reports.py collect data/processed --from \"%USERPROFILE%\\Downloads\"",
              "     and: python gsi_reports.py inventory data/processed   and send gsi_inventory.csv.",
              "While reading each report, fill one row in data/processed/report_findings.csv:",
              "  outcome positive / barren / inconclusive / not_relevant, best_result, sample_type, lon, lat, page_ref.",
              "Most valuable: drilling, trenching or soil grids on farmed / soil-covered ground, positive or barren.",
              "Those are the only tests of the model away from outcrop (the current gold labels are 98% on",
              "uncultivated ground)."]
    (proc / "GSI_FETCH_README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:-10]))
    n_need = sum(r["needed"] == "yes" for r in rows)
    print(f"{len(reps)} reports, {n_need} files needed ({sum(r['needed'] == 'optional' for r in rows)} optional plates)")
    print(f"Written {out} and {proc / 'GSI_FETCH_README.txt'}")


# ----------------------------------------------------------------------------- collect
def _norm(name: str) -> str:
    """'abc (1).xlsx' -> 'abc.xlsx', lower case."""
    stem, dot, ext = name.rpartition(".")
    if name.lower().endswith(".gdb.zip"):
        stem, ext = name[:-8], "gdb.zip"
    stem = re.sub(r"\s*\(\d+\)$", "", stem if dot else name)
    return f"{stem}.{ext}".lower() if dot else stem.lower()


def _kind_from_name(name: str, report_rows: pd.DataFrame) -> str:
    low = name.lower()
    if low.endswith((".pdf", ".docx", ".doc")):
        return "text"
    if low.endswith((".xlsx", ".xls", ".csv")):
        return "tables"
    if low.endswith((".tif", ".tiff", ".jpg", ".png")):
        return "plates"
    if low.endswith(".gdb.zip") or re.search(r"gis|gdb|plate|map|fig", low):
        return "gis"
    if low.endswith(".zip"):   # a plain zip: tables if the report's tables are zipped, else GIS
        t = report_rows[(report_rows["kind"] == "tables")]
        return "tables" if t["file_name"].str.lower().str.endswith(".zip").any() else "gis"
    return "other"


def collect(proc: Path, src: Path) -> None:
    import shutil
    fl_path = proc / "gsi_fetch_list.csv"
    if not fl_path.exists():
        sys.exit("No gsi_fetch_list.csv yet: run  python gsi_reports.py fetchlist data/processed  first")
    if not src.is_dir():
        sys.exit(f"Folder not found: {src}")
    fl = pd.read_csv(fl_path, dtype=str, keep_default_na=False)
    files = [f for f in list(src.iterdir()) + [g for d in src.iterdir() if d.is_dir() for g in d.iterdir()]
             if f.is_file() and not f.name.endswith((".crdownload", ".part", ".tmp"))]
    by_norm = {}
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):   # newest copy wins
        by_norm.setdefault(_norm(f.name), f)
    used, copied = set(), 0
    for i, r in fl.iterrows():
        f = by_norm.get(r["file_name"].lower())
        if f is None:
            continue
        dest = Path(r["save_as"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dest)
            copied += 1
        fl.at[i, "downloaded"] = "yes"
        used.add(f)
    # files the portal may have renamed: they still carry the report id
    ids = sorted(set(fl["report_id"]), key=len, reverse=True)
    extra = 0
    for f in files:
        if f in used:
            continue
        key = re.sub(r"[_\s]", "-", f.name.upper())
        rid = next((x for x in ids if x.upper() in key), None)
        if rid is None:
            continue
        kind = _kind_from_name(f.name, fl[fl["report_id"] == rid])
        dest = Path("data/raw/gsi_reports") / rid / kind / f.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(f, dest)
            extra += 1
        used.add(f)
    fl.to_csv(fl_path, index=False, encoding="utf-8-sig")
    print(f"Looked at {len(files)} files in {src}: {copied} listed files copied into place, "
          f"{extra} renamed files placed by report id.")
    need = fl[fl["needed"] == "yes"]
    for b, g in need.groupby("batch", sort=True):
        print(f"Batch {b}:")
        for rid, gr in g.groupby("report_id", sort=False):
            have_dir = Path("data/raw/gsi_reports") / rid
            n_extra = sum(1 for p in have_dir.rglob("*") if p.is_file() and "_extracted" not in p.parts) if have_dir.exists() else 0
            done = (gr["downloaded"] == "yes").sum()
            status = "complete" if done == len(gr) else (f"{done}/{len(gr)} by name" + (f", {n_extra} files in folder" if n_extra else ""))
            missing = ", ".join(sorted(set(gr.loc[gr["downloaded"] != "yes", "kind"])))
            print(f"  {rid:16} {status}" + (f"   missing: {missing}" if done < len(gr) else ""))
    print("Next: python gsi_reports.py inventory data/processed")


# ----------------------------------------------------------------------------- inventory
def safe_extract(zp: Path, dest: Path) -> None:
    with zipfile.ZipFile(zp) as z:
        root = dest.resolve()
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"unsafe path in {zp.name}: {m.filename}")
        z.extractall(dest)


def numeric_max(series) -> float | None:
    v = pd.to_numeric(series.astype(str).str.replace(r"[<>,\s]", "", regex=True), errors="coerce")
    return float(v.max()) if v.notna().any() else None


def header_row(df: pd.DataFrame) -> int:
    """First of the top 30 rows with at least 3 text cells: the column header."""
    for i in range(min(30, len(df))):
        vals = df.iloc[i].dropna().astype(str).str.strip()
        if (vals.str.match(r"^[A-Za-z]") & (vals.str.len() > 0)).sum() >= 3:
            return i
    return 0


def inventory_table(path: Path, rep: str) -> list[dict]:
    rows = []
    try:
        if path.suffix.lower() == ".csv":
            sheets = {"csv": pd.read_csv(path, header=None, dtype=str, encoding_errors="replace")}
        else:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    except ImportError as e:
        return [{"report_id": rep, "file": path.name, "kind": "table", "error": f"install {e.name} (pip install {e.name})"}]
    except Exception as e:
        return [{"report_id": rep, "file": path.name, "kind": "table", "error": f"{type(e).__name__}: {str(e)[:80]}"}]
    for sh, raw in sheets.items():
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        if raw.empty:
            continue
        h = header_row(raw)
        cols = [str(c).strip().replace("\n", " ") for c in raw.iloc[h].tolist()]
        body = raw.iloc[h + 1:]
        body.columns = [f"{c}#{i}" for i, c in enumerate(cols)]
        au = [c for c in body.columns if AU_COL.search(c.split("#")[0])]
        xy = [c.split("#")[0] for c in body.columns if XY_COL.search(c.split("#")[0])]
        mx = {c.split("#")[0]: numeric_max(body[c]) for c in au}
        rows.append({"report_id": rep, "file": path.name, "kind": "table", "layer_or_sheet": str(sh),
                     "rows": len(body), "header_row": h + 1, "columns": " | ".join(cols)[:400],
                     "coord_columns": ", ".join(xy), "au_columns": ", ".join(c.split("#")[0] for c in au),
                     "au_max": "; ".join(f"{k}={v:g}" for k, v in mx.items() if v is not None)})
    return rows


def list_layers(path: Path) -> list[str]:
    try:
        import pyogrio
        return [str(x[0]) for x in pyogrio.list_layers(path)]
    except Exception:
        import fiona
        return list(fiona.listlayers(path))


def inventory_gis(path: Path, rep: str) -> list[dict]:
    rows = []
    try:
        layers = list_layers(path)
    except Exception as e:
        return [{"report_id": rep, "file": path.name, "kind": "gis", "error": f"{type(e).__name__}: {str(e)[:80]}"}]
    for lyr in layers:
        try:
            g = gpd.read_file(path, layer=lyr)
        except Exception as e:
            rows.append({"report_id": rep, "file": path.name, "kind": "gis", "layer_or_sheet": lyr,
                         "error": f"{type(e).__name__}: {str(e)[:80]}"})
            continue
        fields = [c for c in g.columns if c != "geometry"]
        au = [c for c in fields if AU_COL.search(c)]
        mx = {c: numeric_max(g[c]) for c in au}
        geom = ",".join(sorted(g.geom_type.dropna().unique())) if "geometry" in g else ""
        rows.append({"report_id": rep, "file": path.name, "kind": "gis", "layer_or_sheet": lyr, "rows": len(g),
                     "geometry": geom, "crs": str(g.crs) if g.crs else "", "columns": " | ".join(fields)[:400],
                     "au_columns": ", ".join(au),
                     "au_max": "; ".join(f"{k}={v:g}" for k, v in mx.items() if v is not None)})
    return rows


def inventory(proc: Path, root: Path) -> None:
    if not root.exists():
        sys.exit(f"Nothing to inventory: save report files under {root}/<report_id>/ first")
    rows = []
    for rep_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rep = rep_dir.name
        files = [f for f in rep_dir.rglob("*") if f.is_file() and "_extracted" not in f.parts]
        for f in files:
            low = f.name.lower()
            if low.endswith(".zip"):
                dest = rep_dir / "_extracted" / f.name[:-4]
                if not dest.exists():
                    try:
                        safe_extract(f, dest)
                    except Exception as e:
                        rows.append({"report_id": rep, "file": f.name, "kind": "zip", "error": f"{type(e).__name__}: {e}"})
                        continue
                gdbs = [d for d in dest.rglob("*.gdb") if d.is_dir()]
                for d in gdbs:
                    rows += inventory_gis(d, rep)
                for shp in dest.rglob("*.shp"):
                    rows += inventory_gis(shp, rep)
                for t in list(dest.rglob("*.xls")) + list(dest.rglob("*.xlsx")) + list(dest.rglob("*.csv")):
                    rows += inventory_table(t, rep)
                if not gdbs and not any(dest.rglob("*.shp")) and not any(dest.rglob("*.xls*")):
                    rows.append({"report_id": rep, "file": f.name, "kind": "zip",
                                 "columns": ", ".join(sorted({p.suffix.lower() for p in dest.rglob('*') if p.is_file()}))})
            elif low.endswith((".xlsx", ".xls", ".csv")):
                rows += inventory_table(f, rep)
            elif low.endswith(".shp") or low.endswith(".gpkg"):
                rows += inventory_gis(f, rep)
            elif low.endswith(".pdf"):
                pages = ""
                try:
                    from pypdf import PdfReader
                    pages = len(PdfReader(str(f)).pages)
                except Exception:
                    pass
                rows.append({"report_id": rep, "file": f.name, "kind": "pdf", "rows": pages,
                             "columns": f"{f.stat().st_size / 1e6:.1f} MB"})
        print(f"  {rep}: {len(files)} files")
    keys = ["report_id", "file", "kind", "layer_or_sheet", "rows", "geometry", "crs", "header_row",
            "coord_columns", "au_columns", "au_max", "columns", "error"]
    out = proc / "gsi_inventory.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with_au = [r for r in rows if r.get("au_columns")]
    print(f"\n{len(rows)} tables/layers/files inventoried; {len(with_au)} carry gold columns:")
    for r in with_au[:30]:
        print(f"  {r['report_id']:16} {r['file'][:40]:40} {str(r.get('layer_or_sheet', ''))[:24]:24} "
              f"{r.get('rows', '')} rows  {r['au_columns']}  [{r.get('au_max', '')}]"
              + (f"  coords: {r['coord_columns']}" if r.get("coord_columns") else ""))
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"{len(errs)} could not be read (see the error column), e.g. {errs[0]['file']}: {errs[0]['error']}")
    print(f"Written {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GSI report fetch list and inventory.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fetchlist")
    a.add_argument("processed")
    a.add_argument("--model", default="pg_ens")
    c = sub.add_parser("collect")
    c.add_argument("processed")
    c.add_argument("--from", dest="src", required=True, help="folder the files were downloaded to")
    b = sub.add_parser("inventory")
    b.add_argument("processed")
    b.add_argument("--reports-dir", default="data/raw/gsi_reports")
    args = ap.parse_args()
    if args.cmd == "fetchlist":
        fetchlist(Path(args.processed), args.model)
    elif args.cmd == "collect":
        collect(Path(args.processed), Path(args.src))
    else:
        inventory(Path(args.processed), Path(args.reports_dir))


if __name__ == "__main__":
    main()
