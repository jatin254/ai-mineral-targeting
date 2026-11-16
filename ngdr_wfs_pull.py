#!/usr/bin/env python3
"""
ngdr_wfs_pull.py - catalogue the NGDR WFS (774 feature types) and download the ones that matter here.

ngdr_pull.py found that the geology map address the NGDR API hands out also answers WFS requests,
with 774 feature types, far more than the four documented geology layers. This script:
  1. lists every feature type with its title, keywords and extent  -> wfs_catalog.csv
  2. flags the ones that touch the block AND either name one of our GSI reports or look like
     exploration data (gold, geochemistry, samples, boreholes, trenches, geophysics, ...)
  3. downloads the flagged ones, clipped to the block, as GeoJSON -> wfs/
  4. summarises each download: features, geometry, fields, and any gold-assay-like fields
                                                                       -> wfs_downloads.csv
Run from the repository folder, after ngdr_pull.py (no login needed for this service):
    python ngdr_wfs_pull.py data/processed
    python ngdr_wfs_pull.py data/processed --layers cite:some_layer,cite:other   (download named ones)
Options: --max-layers N (default 80), --no-download (catalogue only), --raw data/raw/ngdr_api
Use of NGDR data is subject to NGDR's terms; credit "Geological Survey of India, NGDR".
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from defusedxml import ElementTree as ET  # type: ignore
except ImportError:
    import xml.etree.ElementTree as ET

AOI = (76.0, 13.75, 77.75, 15.75)
USER_AGENT = "ai-mineral-targeting/1.0"
KEYWORDS = re.compile(r"gold|(^|[_:\s])au([_\s]|$)|geochem|ngcm|stream|soil|bedrock|rock_?chip|sample|assay|"
                      r"bore|drill|trench|pit|explor|block|occurren|mineral|alteration|shear|lineament|vein|"
                      r"gravity|magnet|aero|radiomet|tem\b|ip_|resistiv|working|mine", re.I)
AU_FIELD = re.compile(r"(^|_)(au|gold)(_|$)|au_?pp[bm]|au_?g_?t|gpt|ppb", re.I)
PAUSE_S = 0.3


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child(el, name):
    for c in el:
        if local(c.tag) == name:
            return c
    return None


def text(el, name):
    c = child(el, name)
    return (c.text or "").strip() if c is not None and c.text else ""


def parse_caps(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    out = []
    for ft in (e for e in root.iter() if local(e.tag) == "FeatureType"):
        name = text(ft, "Name")
        if not name:
            continue
        kw = []
        k = child(ft, "Keywords")
        if k is None:
            k = child(ft, "KeywordList")
        if k is not None:
            kw = [w.text.strip() for w in k if w.text and w.text.strip()] or [x.strip() for x in (k.text or "").split(",") if x.strip()]
        bbox = None
        llb = child(ft, "LatLongBoundingBox")
        if llb is not None:
            try:
                bbox = tuple(float(llb.attrib[a]) for a in ("minx", "miny", "maxx", "maxy"))
            except (KeyError, ValueError):
                bbox = None
        wgs = child(ft, "WGS84BoundingBox")
        if wgs is not None and bbox is None:
            try:
                lo = [float(v) for v in text(wgs, "LowerCorner").split()]
                hi = [float(v) for v in text(wgs, "UpperCorner").split()]
                bbox = (lo[0], lo[1], hi[0], hi[1])
            except (ValueError, IndexError):
                bbox = None
        out.append({"name": name, "title": text(ft, "Title"), "abstract": text(ft, "Abstract")[:300],
                    "keywords": "; ".join(kw)[:300], "bbox": bbox})
    return out


def intersects(b, aoi=AOI):
    return b is not None and not (b[2] < aoi[0] or b[0] > aoi[2] or b[3] < aoi[1] or b[1] > aoi[3])


def report_ids(proc: Path) -> set[str]:
    ids = set()
    for p in list(proc.glob("models/*/step4/report_pull_list.csv")) + [proc / "report_findings.csv"]:
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                ids |= {r["report_id"] for r in csv.DictReader(f) if r.get("report_id")}
    try:
        import geopandas as gpd
        b = gpd.read_file(proc / "ka_ap_v1.gpkg", layer="exploration_blocks", ignore_geometry=True)
        ids |= set(b["report_id"].astype(str))
    except Exception:
        pass
    return ids


def id_patterns(ids: set[str]) -> dict[str, re.Pattern]:
    """'SRO-11785-1987' matches sro_11785_1987, SRO11785-1987, 11785_1987 ..."""
    pats = {}
    for rid in ids:
        m = re.match(r"([A-Za-z]+)-?(\d+)-?(\d{4})", rid)
        if m:
            pats[rid] = re.compile(rf"{m.group(2)}[\W_]*{m.group(3)}", re.I)
    return pats


def get(url: str, timeout=180, ctx=None) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        data, ctype = r.read(), r.headers.get("Content-Type", "")
    time.sleep(PAUSE_S)
    return data, ctype


def with_params(base: str, **params) -> str:
    return base + ("&" if "?" in base else "?") + urllib.parse.urlencode(params)


def summarise(js: dict) -> dict:
    feats = js.get("features", [])
    geoms = sorted({(f.get("geometry") or {}).get("type", "None") for f in feats})
    fields = []
    for f in feats[:200]:
        fields += [k for k in (f.get("properties") or {}) if k not in fields]
    au = [k for k in fields if AU_FIELD.search(k)]
    au_max = {}
    for k in au:
        vals = []
        for f in feats:
            try:
                vals.append(float((f.get("properties") or {}).get(k)))
            except (TypeError, ValueError):
                pass
        if vals:
            au_max[k] = max(vals)
    return {"features": len(feats), "geometry": ",".join(geoms), "n_fields": len(fields),
            "fields": ", ".join(fields[:40]), "au_fields": ", ".join(au),
            "au_max": json.dumps(au_max) if au_max else ""}


def main() -> None:
    ap = argparse.ArgumentParser(description="Catalogue and pull the NGDR WFS for the block.")
    ap.add_argument("processed", help="data/processed (report ids are read from here)")
    ap.add_argument("--raw", default="data/raw/ngdr_api", help="folder ngdr_pull.py wrote")
    ap.add_argument("--url", default="", help="WFS address (default: from raw/geologylayer.json)")
    ap.add_argument("--layers", default="", help="comma-separated feature types to download regardless of flags")
    ap.add_argument("--max-layers", type=int, default=80)
    ap.add_argument("--max-features", type=int, default=100000)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="download again layers already saved in raw/wfs")
    ap.add_argument("--ca-bundle", default=None)
    args = ap.parse_args()
    proc, raw = Path(args.processed), Path(args.raw)
    (raw / "wfs").mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=args.ca_bundle) if args.ca_bundle else ssl.create_default_context()

    url = args.url
    if not url:
        try:
            gl = json.loads((raw / "raw" / "geologylayer.json").read_text(encoding="utf-8"))
            url = next(str(e.get("url", "")).strip() for e in gl if e.get("url"))
        except Exception:
            sys.exit("No WFS address: run ngdr_pull.py first, or pass --url")
    caps_path = raw / "raw" / "wfs_capabilities.xml"
    if caps_path.exists():
        caps = caps_path.read_bytes()
    else:
        try:
            caps, _ = get(with_params(url, SERVICE="WFS", REQUEST="GetCapabilities", VERSION="1.0.0"), ctx=ctx)
            caps_path.write_bytes(caps)
        except Exception as ex:
            sys.exit(f"WFS GetCapabilities failed: {type(ex).__name__}: {ex}")
    fts = parse_caps(caps)
    pats = id_patterns(report_ids(proc))
    for f in fts:
        hay = f"{f['name']} {f['title']} {f['abstract']} {f['keywords']}"
        f["reports"] = " ".join(rid for rid, p in pats.items() if p.search(hay))
        f["keyword_hit"] = bool(KEYWORDS.search(hay))
        f["in_block"] = "unknown" if f["bbox"] is None else intersects(f["bbox"])
    print(f"WFS {url}: {len(fts)} feature types; {sum(f['in_block'] is True for f in fts)} touch the block, "
          f"{sum(f['in_block'] == 'unknown' for f in fts)} without an extent, "
          f"{sum(bool(f['reports']) for f in fts)} name one of our GSI reports")

    with open(raw / "wfs_catalog.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "title", "in_block", "reports", "keyword_hit", "bbox", "keywords", "abstract"])
        w.writeheader()
        for f in fts:
            w.writerow({**f, "bbox": "" if f["bbox"] is None else ",".join(f"{v:.4f}" for v in f["bbox"])})

    forced = [x.strip() for x in args.layers.split(",") if x.strip()]
    flagged = [f for f in fts if f["in_block"] is not False and (f["reports"] or f["keyword_hit"])]
    # report layers first, then layers inside the block, then the rest
    flagged.sort(key=lambda f: (not f["reports"], f["in_block"] is not True, f["name"]))
    todo = forced + [f["name"] for f in flagged if f["name"] not in forced][:max(0, args.max_layers - len(forced))]
    print(f"Flagged for download: {len(flagged)} (taking {len(todo)}, --max-layers {args.max_layers})")
    if args.no_download:
        print(f"Catalogue written: {raw / 'wfs_catalog.csv'}")
        return

    rows = []
    for i, name in enumerate(todo, 1):
        q = dict(SERVICE="WFS", REQUEST="GetFeature", VERSION="1.0.0", TYPENAME=name, SRSNAME="EPSG:4326",
                 BBOX=",".join(map(str, AOI)), OUTPUTFORMAT="application/json", MAXFEATURES=args.max_features)
        row = {"name": name}
        meta = next((f for f in fts if f["name"] == name), {})
        row.update({"title": meta.get("title", ""), "reports": meta.get("reports", "")})
        saved = raw / "wfs" / (re.sub(r"[^\w.-]", "_", name) + ".geojson")
        if saved.exists() and not args.refresh:   # fetched by an earlier run: summarise from disk
            try:
                js = json.loads(saved.read_text(encoding="utf-8"))
                row.update(summarise(js))
                row.update({"file": saved.name, "note": "already downloaded"})
                rows.append(row)
                extra = f"  gold fields: {row['au_fields']}" if row.get("au_fields") else ""
                print(f"  [{i}/{len(todo)}] {name}: {row.get('features', 0)} features (already downloaded){extra}", flush=True)
                continue
            except (json.JSONDecodeError, OSError):
                pass
        try:
            data, ctype = get(with_params(url, **q), timeout=600, ctx=ctx)
            if data.lstrip()[:1] == b"<":
                msg = re.sub(r"<[^>]+>", " ", data[:1500].decode("utf-8", "replace"))
                row["error"] = re.sub(r"\s+", " ", msg).strip()[:150]
            else:
                js = json.loads(data)
                row.update(summarise(js))
                if row["features"]:
                    path = raw / "wfs" / (re.sub(r"[^\w.-]", "_", name) + ".geojson")
                    path.write_text(json.dumps(js), encoding="utf-8")
                    row["file"] = path.name
                if row["features"] >= args.max_features:
                    row["note"] = "hit --max-features: may be truncated"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as ex:
            row["error"] = f"{type(ex).__name__}: {str(ex)[:100]}"
        rows.append(row)
        tag = f"{row.get('features', 0)} features" if "error" not in row else f"error: {row['error'][:60]}"
        extra = f"  gold fields: {row['au_fields']}" if row.get("au_fields") else ""
        print(f"  [{i}/{len(todo)}] {name}: {tag}{extra}", flush=True)

    keys = ["name", "title", "reports", "features", "geometry", "au_fields", "au_max", "n_fields", "fields", "file", "note", "error"]
    with open(raw / "wfs_downloads.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    got = [r for r in rows if r.get("features")]
    au = [r for r in got if r.get("au_fields")]
    print(f"\nDownloaded {len(got)} layers with features inside the block; {len(au)} carry gold-assay-like fields:")
    for r in au[:20]:
        print(f"  {r['name']}: {r['features']} features, {r['au_fields']}  max {r.get('au_max', '')}")
    print(f"Outputs: {raw / 'wfs_catalog.csv'}, {raw / 'wfs_downloads.csv'}, {raw / 'wfs'}")


if __name__ == "__main__":
    main()
