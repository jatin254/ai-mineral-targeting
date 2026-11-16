#!/usr/bin/env python3
"""
inventory.py - inventory a folder of raw geoscience datasets and draft a dataset register.

Run from the repository folder:
    python inventory.py data/raw/aikosh-gsi-ka-ap --source "AIKosh / GSI" --licence "NA on AIKosh (unconfirmed)"
    python inventory.py data/raw/aikosh-gsi-ka-ap --hash    # also SHA256 every file (slower: ASTER is ~6.6 GB)

Outputs (written to data/register/ when the input folder sits under data/raw/):
    inventory_<source-folder>.csv   one row per file / layer / sub-dataset
    register_<source-folder>.csv    one row per dataset (top-level folder), with blank columns to finish by hand

Install (conda-forge recommended on Windows; its GDAL has more format drivers than pip wheels):
    conda create -n te -c conda-forge python=3.11 rasterio pyogrio pyproj openpyxl
    conda activate te

The script only reads. It never modifies anything under the input folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import pyogrio
    import rasterio
    from pyproj import CRS, Transformer
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. See the install line at the top of this file.")

VECTOR_EXT = {".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".gml", ".tab", ".mif"}
RASTER_EXT = {".tif", ".tiff", ".img", ".ers", ".asc", ".grd", ".hdf", ".h5", ".he5",
              ".jp2", ".vrt", ".bil", ".bsq", ".bip", ".nc", ".dem"}
TABLE_EXT = {".csv", ".tsv", ".txt", ".xyz", ".dat"}
SHEET_EXT = {".xlsx", ".xlsm", ".xls"}
DOC_EXT = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".rtf"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
ARCHIVE_EXT = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz"}
SIDECAR_EXT = {".shx", ".prj", ".cpg", ".sbn", ".sbx", ".qix", ".fbn", ".fbx", ".ain", ".aih",
               ".atx", ".ixs", ".mxs", ".ovr", ".aux", ".rrd", ".tfw", ".tifw", ".jgw", ".pgw",
               ".hdr", ".lock", ".qmd", ".qml", ".sld", ".lyr", ".lyrx", ".gi", ".xml", ".enp"}

COLUMNS = ["dataset", "rel_path", "layer", "ext", "kind", "driver", "crs", "geom_or_dtype",
           "features_or_bands", "width", "height", "resolution", "nodata", "bbox_wgs84",
           "fields", "size_mb", "sha256", "status", "notes"]

GRD_HINT = ("GDAL can't open it; likely a Geosoft grid. Use a GeoTIFF/ASCII/ERS copy if the "
            "dataset has one, or export from Oasis montaj.")
GEOSOFT_GDB_HINT = "Geosoft database (proprietary). Use an XYZ/CSV export if the dataset has one."


# ----------------------------------------------------------------------------- helpers

def crs_label(crs_in) -> str:
    if not crs_in:
        return ""
    try:
        c = CRS.from_user_input(crs_in)
        epsg = c.to_epsg()
        return f"EPSG:{epsg}" if epsg else c.name
    except Exception:
        return str(crs_in)[:60]


def bbox_wgs84(crs_in, bounds):
    """Return (minlon, minlat, maxlon, maxlat) or None."""
    if not crs_in or bounds is None:
        return None
    try:
        b = [float(v) for v in bounds]
        if len(b) != 4 or any(math.isnan(v) or math.isinf(v) for v in b):
            return None
        t = Transformer.from_crs(CRS.from_user_input(crs_in), CRS.from_epsg(4326), always_xy=True)
        return tuple(round(v, 4) for v in t.transform_bounds(*b))
    except Exception:
        return None


def fmt_bbox(b) -> str:
    return "" if not b else ",".join(f"{v:.4f}" for v in b)


def union_bbox(a, b):
    if not a:
        return b
    if not b:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    total = 0
    for dp, _, fns in os.walk(path):
        for fn in fns:
            try:
                total += (Path(dp) / fn).stat().st_size
            except OSError:
                pass
    return total


# ----------------------------------------------------------------------------- readers

def _raster_meta(ds, layer: str = "") -> dict:
    crs = ds.crs.to_wkt() if ds.crs else None
    if ds.crs is None:
        units = "px"
    elif ds.crs.is_geographic:
        units = "deg"
    else:
        units = getattr(ds.crs, "linear_units", "") or "units"
    return {
        "kind": "raster", "layer": layer, "driver": ds.driver, "crs": crs_label(crs),
        "geom_or_dtype": ds.dtypes[0] if ds.count else "", "features_or_bands": ds.count,
        "width": ds.width, "height": ds.height,
        "resolution": f"{ds.res[0]:.6g} x {ds.res[1]:.6g} {units}",
        "nodata": "" if ds.nodata is None else ds.nodata,
        "bbox_wgs84": bbox_wgs84(crs, ds.bounds),
        "status": "ok" if ds.crs else "no CRS",
    }


def raster_rows(path: Path) -> list[dict]:
    with rasterio.open(path) as ds:
        subs = list(ds.subdatasets or [])
        if subs and ds.count == 0:
            rows = []
            for sd in subs:
                name = sd.split(":")[-1]
                try:
                    with rasterio.open(sd) as sds:
                        rows.append(_raster_meta(sds, layer=name))
                except Exception as e:
                    rows.append({"kind": "raster", "layer": name, "status": "unreadable",
                                 "notes": f"{type(e).__name__}: {str(e)[:150]}"})
            return rows
        return [_raster_meta(ds)]


def vector_rows(path: Path) -> list[dict]:
    layers = [None]
    try:
        lyrs = pyogrio.list_layers(path)
        if len(lyrs) > 1:
            layers = [str(l[0]) for l in lyrs]
    except Exception:
        pass
    rows = []
    for lyr in layers:
        try:
            try:
                info = pyogrio.read_info(path, layer=lyr, force_feature_count=True,
                                         force_total_bounds=True)
            except TypeError:  # older pyogrio
                info = pyogrio.read_info(path, layer=lyr)
            crs = info.get("crs")
            fields = [str(f) for f in info.get("fields", [])]
            rows.append({
                "kind": "vector", "layer": lyr or "", "driver": info.get("driver", ""),
                "crs": crs_label(crs), "geom_or_dtype": info.get("geometry_type", ""),
                "features_or_bands": info.get("features", ""),
                "bbox_wgs84": bbox_wgs84(crs, info.get("total_bounds")),
                "fields": ";".join(fields)[:500],
                "status": "ok" if crs else "no CRS",
            })
        except Exception as e:
            rows.append({"kind": "vector", "layer": lyr or "", "status": "unreadable",
                         "notes": f"{type(e).__name__}: {str(e)[:150]}"})
    return rows


def table_rows(path: Path) -> list[dict]:
    n, header = 0, ""
    with open(path, "rb") as f:
        for line in f:
            if not header and line.strip():
                header = line.decode("utf-8", "replace").strip()
            n += 1
    notes = []
    if header.startswith("/"):
        notes.append("Geosoft XYZ-style header ('/' comment lines)")
    tokens = [t.strip().strip('"').lower() for t in header.replace(";", ",").replace("\t", ",").split(",")]
    coord_names = {"x", "y", "lon", "long", "longitude", "lat", "latitude", "easting", "northing",
                   "east", "north", "utm_e", "utm_n", "x_utm", "y_utm"}
    coords = [t for t in tokens if t in coord_names]
    if coords:
        notes.append("coordinate columns: " + ",".join(coords))
    return [{"kind": "table", "features_or_bands": n, "fields": header[:500],
             "status": "ok", "notes": "; ".join(notes)}]


def sheet_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xls":
        return [{"kind": "table", "status": "ok", "notes": "legacy .xls, not inspected"}]
    try:
        from openpyxl import load_workbook
    except ImportError:
        return [{"kind": "table", "status": "ok", "notes": "openpyxl not installed, not inspected"}]
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = []
    for ws in wb.worksheets:
        hdr = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ()) or ()
        rows.append({"kind": "table", "layer": ws.title, "features_or_bands": ws.max_row,
                     "fields": ";".join(str(h) for h in hdr if h is not None)[:500],
                     "status": "ok"})
    wb.close()
    return rows


# ----------------------------------------------------------------------------- classification

def classify(path: Path, shp_stems: set[str]) -> str:
    name, ext = path.name.lower(), path.suffix.lower()
    if ext == ".dbf":
        return "sidecar" if path.stem.lower() in shp_stems else "table_dbf"
    if ext == ".gdb":
        return "geosoft_gdb"
    if ext in VECTOR_EXT:
        return "vector"
    if ext in RASTER_EXT:
        return "raster"
    if ext in TABLE_EXT:
        return "table"
    if ext in SHEET_EXT:
        return "sheet"
    if ext in SIDECAR_EXT or name.endswith(".aux.xml"):
        return "sidecar"
    if ext in DOC_EXT:
        return "document"
    if ext in IMAGE_EXT:
        return "image"
    if ext in ARCHIVE_EXT:
        return "archive"
    return "unknown"


def describe(path: Path, kind: str) -> list[dict]:
    try:
        if kind == "raster":
            try:
                return raster_rows(path)
            except Exception as e:
                hint = GRD_HINT if path.suffix.lower() == ".grd" else ""
                return [{"kind": "raster", "status": "unreadable",
                         "notes": f"{hint} [{type(e).__name__}: {str(e)[:120]}]".strip()}]
        if kind in ("vector", "table_dbf"):
            return vector_rows(path)
        if kind == "table":
            return table_rows(path)
        if kind == "sheet":
            return sheet_rows(path)
        if kind == "geosoft_gdb":
            return [{"kind": "database", "status": "needs conversion", "notes": GEOSOFT_GDB_HINT}]
        if kind == "archive":
            return [{"kind": "archive", "status": "not extracted",
                     "notes": "Extract it so its contents get inventoried."}]
        if kind == "unknown":
            # GDAL may still know it (e.g. extensionless ENVI data next to a .hdr)
            for reader in (raster_rows, vector_rows):
                try:
                    rows = reader(path)
                    if rows and rows[0].get("status") != "unreadable":
                        return rows
                except Exception:
                    continue
            return [{"kind": "other", "status": "unrecognised"}]
        return [{"kind": kind, "status": "ok"}]  # sidecar / document / image
    except Exception as e:
        return [{"kind": kind, "status": "unreadable",
                 "notes": f"{type(e).__name__}: {str(e)[:150]}"}]


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Inventory raw geoscience datasets.")
    ap.add_argument("root", help="folder to inventory, e.g. data/raw/aikosh-gsi-ka-ap")
    ap.add_argument("--hash", action="store_true", help="compute SHA256 for every file")
    ap.add_argument("--out", help="output folder (default: data/register/)")
    ap.add_argument("--source", default="", help="prefill the register's source column")
    ap.add_argument("--licence", default="", help="prefill the register's licence column")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a folder: {root}")
    if args.out:
        out = Path(args.out)
    elif root.parent.name.lower() == "raw":
        out = root.parent.parent / "register"
    else:
        out = root.parent / "register"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        dirnames.sort()
        # Directories that GDAL treats as a single dataset: ESRI FileGDB, ArcInfo binary grid
        for dn in list(dirnames):
            sub = d / dn
            if dn.lower().endswith(".gdb"):
                kind, reader = "vector", vector_rows
            elif (sub / "hdr.adf").exists():
                kind, reader = "raster", raster_rows
            else:
                continue
            dirnames.remove(dn)
            try:
                described = reader(sub)
            except Exception as e:
                described = [{"kind": kind, "status": "unreadable",
                              "notes": f"{type(e).__name__}: {str(e)[:150]}"}]
            size = dir_size(sub)
            rel = sub.relative_to(root)
            for r in described:
                rows.append({"dataset": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                             "rel_path": str(rel), "ext": "<dir>", "size_mb": round(size / 1e6, 3),
                             **r})
            n_files += 1

        shp_stems = {Path(f).stem.lower() for f in filenames if f.lower().endswith(".shp")}
        for fn in sorted(filenames):
            p = d / fn
            n_files += 1
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            kind = classify(p, shp_stems)
            digest = ""
            if args.hash:
                try:
                    digest = sha256(p)
                except OSError as e:
                    digest = f"error: {e}"
            rel = p.relative_to(root)
            base = {"dataset": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                    "rel_path": str(rel), "ext": p.suffix.lower(),
                    "size_mb": round(size / 1e6, 3), "sha256": digest}
            for r in describe(p, kind):
                rows.append({**base, **r})
            if n_files % 200 == 0:
                print(f"  ...{n_files} files scanned", flush=True)

    # --- inventory CSV
    inv_path = out / f"inventory_{root.name}.csv"
    with open(inv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in COLUMNS}, **r, "bbox_wgs84": fmt_bbox(r.get("bbox_wgs84"))})

    # --- register CSV (one row per dataset)
    reg = defaultdict(lambda: {"files": set(), "size": 0.0, "kinds": Counter(), "crs": set(),
                               "drivers": set(), "bbox": None, "unreadable": 0, "no_crs": 0,
                               "conversion": 0})
    for r in rows:
        g = reg[r["dataset"]]
        if r["rel_path"] not in g["files"]:
            g["files"].add(r["rel_path"])
            g["size"] += r["size_mb"]
        if r["kind"] == "sidecar":
            continue
        g["kinds"][r["kind"]] += 1
        if r.get("crs"):
            g["crs"].add(r["crs"])
        if r.get("driver"):
            g["drivers"].add(r["driver"])
        g["bbox"] = union_bbox(g["bbox"], r.get("bbox_wgs84"))
        st = r.get("status", "")
        g["unreadable"] += st in ("unreadable", "unrecognised")
        g["no_crs"] += st == "no CRS"
        g["conversion"] += st in ("needs conversion", "not extracted")

    reg_cols = ["dataset", "files", "size_mb", "data_items", "formats", "crs", "bbox_wgs84",
                "issues", "source", "source_url", "licence", "downloaded_on", "inventoried_on",
                "notes"]
    reg_rows = []
    for name in sorted(reg):
        g = reg[name]
        issues = []
        if len(g["crs"]) > 1:
            issues.append(f"mixed CRS ({len(g['crs'])})")
        if g["no_crs"]:
            issues.append(f"{g['no_crs']} without CRS")
        if g["unreadable"]:
            issues.append(f"{g['unreadable']} unreadable")
        if g["conversion"]:
            issues.append(f"{g['conversion']} need conversion/extraction")
        reg_rows.append({
            "dataset": name, "files": len(g["files"]), "size_mb": round(g["size"], 1),
            "data_items": "; ".join(f"{k}:{v}" for k, v in sorted(g["kinds"].items())),
            "formats": "; ".join(sorted(g["drivers"])), "crs": "; ".join(sorted(g["crs"])),
            "bbox_wgs84": fmt_bbox(g["bbox"]), "issues": "; ".join(issues),
            "source": args.source, "source_url": "", "licence": args.licence,
            "downloaded_on": "", "inventoried_on": date.today().isoformat(), "notes": "",
        })
    reg_path = out / f"register_{root.name}.csv"
    with open(reg_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=reg_cols)
        w.writeheader()
        w.writerows(reg_rows)

    # --- console summary
    print(f"\nScanned {n_files} files in {root}\n")
    print(f"{'Dataset':32} {'Files':>6} {'Size MB':>10}  {'CRS':28} Issues")
    print("-" * 100)
    all_crs, extent = Counter(), None
    for rr in reg_rows:
        g = reg[rr["dataset"]]
        all_crs.update(g["crs"])
        extent = union_bbox(extent, g["bbox"])
        crs_txt = rr["crs"] if len(rr["crs"]) <= 28 else rr["crs"][:25] + "..."
        print(f"{rr['dataset'][:32]:32} {rr['files']:>6} {rr['size_mb']:>10.1f}  "
              f"{crs_txt:28} {rr['issues'] or '-'}")
    print("\nCRSs across all datasets:", ", ".join(sorted(all_crs)) or "none found")
    print("Combined extent (lon/lat):", fmt_bbox(extent) or "n/a")
    bad = [r for r in rows if r.get("status") not in ("ok",) and r["kind"] != "sidecar"]
    if bad:
        print(f"\nItems needing attention ({len(bad)}; first 20 shown):")
        for r in bad[:20]:
            print(f"  [{r.get('status')}] {r['rel_path']} {r.get('layer', '')} {r.get('notes', '')}"[:160])
    print(f"\nWrote:\n  {inv_path}\n  {reg_path}")


if __name__ == "__main__":
    main()
