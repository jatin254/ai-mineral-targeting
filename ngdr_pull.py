#!/usr/bin/env python3
"""
ngdr_pull.py - pull what the NGDR API (via API Setu) offers for the Karnataka/Andhra block.

Based on "NGDR Geo Spatial API Sharing guideline using API Setu" (GSI/BISAG-N). That API has four
endpoints, all at https://geodataindia.gov.in/ngdrapi :
    POST /authenticate                          username + password -> bearer token
    POST /api/getcommodity/{commodity}          commodity records (commodity, states, districts)
    POST /api/getstatedistict/{state},{district} commodity records for one district
    POST /api/geologylayer                      names + WMS address of the 1:2M geology layers
What it does NOT offer: the exploration reports, report tables, borehole or geochemistry files.
Those are downloaded from the NGDR portal itself (geodataindia.gov.in, login + NDA); use
models/<model>/step4/report_pull_list.csv and data/processed/report_findings.csv for that.

Run from the repository folder:
    set NGDR_USER=your_ngdr_username
    set NGDR_PASSWORD=your_ngdr_password
    python ngdr_pull.py data/processed --model pg_ens
(or leave them unset and you will be asked; credentials and the token are never written to disk)

Writes data/raw/ngdr_api/
    commodity_<name>.csv          every record for each --commodities entry (default Gold), with
                                  in_block = whether its states/districts touch this block
    district_commodities.csv      all commodity records for the block's districts
    wms_layers.csv                every layer the geology WMS publishes (from GetCapabilities)
    wms_maps/<layer>.tif (+.png)  the block rendered by each geology layer, georeferenced (EPSG:4326),
                                  for visual overlay in QGIS. These are pictures of the map, not data.
    wms_featureinfo.csv           what each queryable layer says at every prospect peak
    wfs/<layer>.geojson           vector features, only if the server also offers WFS
    raw/                          the untouched JSON / XML replies
    ngdr_pull_report.txt          what worked, what did not, and what each output is good for
Use of NGDR data is subject to NGDR's terms; credit "Geological Survey of India, NGDR" in any output.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:  # safer XML parsing if available; the replies come from an outside server
    from defusedxml import ElementTree as ET  # type: ignore
except ImportError:
    import xml.etree.ElementTree as ET

API_BASE = "https://geodataindia.gov.in/ngdrapi"
USER_AGENT = "ai-mineral-targeting/1.0"
AOI = (76.0, 13.75, 77.75, 15.75)          # lon/lat of the block
STATES = {
    "KARNATAKA": {
        "Chitradurga": ["Chitradurga"],
        "Davanagere": ["Davanagere", "Davangere"],
        "Tumakuru": ["Tumakuru", "Tumkur"],
        "Ballari": ["Ballari", "Bellary"],
        "Vijayanagara": ["Vijayanagara", "Vijayanagar"],
        "Koppal": ["Koppal"],
        "Raichur": ["Raichur"],
        "Shivamogga": ["Shivamogga", "Shimoga"],
        "Chikkamagaluru": ["Chikkamagaluru", "Chikmagalur"],
        "Haveri": ["Haveri"],
    },
    "ANDHRA PRADESH": {
        "Anantapur": ["Anantapur", "Anantapuramu"],
        "Sri Sathya Sai": ["Sri Sathya Sai", "Sri Sathyasai"],
        "Kurnool": ["Kurnool"],
    },
}
LAYER_KEYWORDS = re.compile(r"geolog|litho|fault|thrust|shear|lineament|commodit|mineral|gold|dyke|geochem|"
                            r"gravity|magnet|radiomet|aero", re.I)
PAUSE_S = 0.3


class NgdrError(RuntimeError):
    pass


class Client:
    def __init__(self, base: str, user: str, password: str, ca_bundle: str | None):
        self.base = base.rstrip("/")
        self.user, self._pw = user, password
        self.ctx = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else ssl.create_default_context()
        self.token = None

    def _open(self, url, data=None, headers=None, timeout=90):
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
                return r.read(), r.headers.get("Content-Type", "")
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise NgdrError("TLS certificate check failed. Try: pip install certifi, then rerun with "
                                "--ca-bundle followed by the path printed by: python -c \"import certifi;print(certifi.where())\"")
            raise

    def login(self):
        body = json.dumps({"username": self.user, "password": self._pw}).encode()
        try:
            raw, _ = self._open(f"{self.base}/authenticate", body,
                                {"Content-Type": "application/json", "accept": "application/json"})
        except urllib.error.HTTPError as e:
            raise NgdrError(f"login refused (HTTP {e.code}): check NGDR_USER / NGDR_PASSWORD") from None
        tok = json.loads(raw).get("token")
        if not tok:
            raise NgdrError("login reply had no token")
        self.token = tok

    def post(self, path: str):
        """POST an API path with the bearer token; logs in again once if the token has expired."""
        for attempt in (1, 2):
            if self.token is None:
                self.login()
            try:
                raw, _ = self._open(f"{self.base}{path}", b"",
                                    {"accept": "application/json", "Authorization": f"Bearer {self.token}"})
                time.sleep(PAUSE_S)
                return json.loads(raw) if raw.strip() else []
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and attempt == 1:
                    self.token = None
                    continue
                if e.code == 404:
                    return []
                raise NgdrError(f"{path}: HTTP {e.code}") from None
        return []

    def get(self, url: str, auth: bool = False, timeout=120):
        headers = {"Authorization": f"Bearer {self.token}"} if auth and self.token else {}
        raw, ctype = self._open(url, None, headers, timeout)
        time.sleep(PAUSE_S)
        return raw, ctype


# ----------------------------------------------------------------------------- helpers
def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def as_records(js) -> list[dict]:
    if isinstance(js, dict):
        for k in ("data", "result", "results", "records"):
            if isinstance(js.get(k), list):
                return js[k]
        return [js]
    return [r if isinstance(r, dict) else {"value": r} for r in (js or [])]


def in_block(rec: dict) -> bool:
    st = str(rec.get("state_name", "")).upper()
    ds = str(rec.get("district_name", "")).lower()
    for state, dists in STATES.items():
        if state in st and any(v.lower() in ds for vs in dists.values() for v in vs):
            return True
    return False


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_all(el, name):
    return [e for e in el.iter() if local(e.tag) == name]


def child_text(el, name):
    for c in el:
        if local(c.tag) == name:
            return (c.text or "").strip()
    return ""


def parse_wms_caps(xml: bytes) -> tuple[list[dict], list[str], list[str], str]:
    root = ET.fromstring(xml)
    version = root.attrib.get("version", "")
    layers = []
    for lyr in find_all(root, "Layer"):
        name = child_text(lyr, "Name")
        if not name:
            continue
        bbox = ""
        for c in lyr:
            if local(c.tag) == "LatLonBoundingBox":
                a = c.attrib
                bbox = f"{a.get('minx')},{a.get('miny')},{a.get('maxx')},{a.get('maxy')}"
            elif local(c.tag) == "EX_GeographicBoundingBox":
                bbox = ",".join(child_text(c, k) for k in ("westBoundLongitude", "southBoundLatitude",
                                                            "eastBoundLongitude", "northBoundLatitude"))
        layers.append({"name": name, "title": child_text(lyr, "Title"), "abstract": child_text(lyr, "Abstract")[:300],
                       "queryable": lyr.attrib.get("queryable", "0") in ("1", "true"), "bbox_lonlat": bbox})

    def formats(req):
        out = []
        for r in find_all(root, req):
            out += [(f.text or "").strip() for f in r if local(f.tag) == "Format"]
        return out
    return layers, formats("GetMap"), formats("GetFeatureInfo"), version


def wms_url(base: str, **params) -> str:
    sep = "&" if "?" in base else "?"
    return base + sep + urllib.parse.urlencode(params)


def is_xml_error(raw: bytes, ctype: str) -> str:
    if "xml" in ctype or raw[:200].lstrip().startswith(b"<"):
        txt = raw[:2000].decode("utf-8", "replace")
        m = re.search(r"<(?:\w+:)?ServiceException[^>]*>(.*?)</", txt, re.S)
        if m or "Exception" in txt:
            return (m.group(1) if m else txt).strip()[:200]
    return ""


def prospects_table(proc: Path, model: str) -> list[dict]:
    for p in (proc / "models" / model / "step4" / "prospects_after_check.csv", proc / "models" / model / "prospects.csv"):
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            tier = "tier_after" if "tier_after" in rows[0] else "best_tier"
            return [{"prospect": r["prospect"], "tier": r[tier], "lon": float(r["peak_lon"]), "lat": float(r["peak_lat"])}
                    for r in rows if r.get(tier) not in ("dropped", "") and r.get("peak_lon")]
    return []


def png_to_geotiff(png: bytes, tif: Path, bbox, width, height) -> bool:
    try:
        import numpy as np
        import rasterio
        from PIL import Image
        from rasterio.enums import ColorInterp
        from rasterio.transform import from_bounds
    except ImportError:
        return False
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    prof = dict(driver="GTiff", width=width, height=height, count=4, dtype="uint8", crs="EPSG:4326",
                transform=from_bounds(*bbox, width, height), compress="deflate", photometric="RGB", alpha="YES")
    with rasterio.open(tif, "w", **prof) as dst:
        dst.write(np.transpose(img, (2, 0, 1)))
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    return True


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Pull NGDR API data for the block.")
    ap.add_argument("processed", help="data/processed (prospect peaks are read from the model folder)")
    ap.add_argument("--model", default="pg_ens")
    ap.add_argument("--out", default="data/raw/ngdr_api")
    ap.add_argument("--commodities", default="Gold", help="comma-separated, e.g. Gold,Copper,Tungsten")
    ap.add_argument("--all-layers", action="store_true", help="render every WMS layer, not only geology ones")
    ap.add_argument("--map-px", type=int, default=1750, help="width of each WMS map in pixels")
    ap.add_argument("--api-base", default=API_BASE)
    ap.add_argument("--ca-bundle", default=None, help="CA certificate file, if the TLS check fails")
    args = ap.parse_args()
    proc, out = Path(args.processed), Path(args.out)
    for d in ("raw", "wms_maps", "wfs"):
        (out / d).mkdir(parents=True, exist_ok=True)
    report = []

    def note(msg):
        print(msg, flush=True)
        report.append(msg)

    user = os.environ.get("NGDR_USER") or input("NGDR username: ").strip()
    pw = os.environ.get("NGDR_PASSWORD") or getpass.getpass("NGDR password (not shown, not saved): ")
    cli = Client(args.api_base, user, pw, args.ca_bundle)
    try:
        cli.login()
    except (NgdrError, urllib.error.URLError) as e:
        sys.exit(f"Could not log in to NGDR: {e}")
    note("Logged in to the NGDR API.")

    # 1. commodity records
    for com in [c.strip() for c in args.commodities.split(",") if c.strip()]:
        try:
            recs = as_records(cli.post("/api/getcommodity/" + urllib.parse.quote(com)))
        except NgdrError as e:
            note(f"getcommodity/{com}: {e}")
            continue
        (out / "raw" / f"commodity_{com}.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
        for r in recs:
            r["in_block"] = in_block(r)
        write_csv(out / f"commodity_{com}.csv", recs)
        nb = sum(r["in_block"] for r in recs)
        note(f"{com}: {len(recs)} records India-wide, {nb} touching this block's districts "
             f"(fields: {', '.join(k for k in (recs[0] if recs else {}) if k != 'in_block')})")

    # 2. district commodity lists
    drows, seen = [], set()
    for state, dists in STATES.items():
        for dname, variants in dists.items():
            got, used = [], ""
            for v in variants:
                try:
                    got = as_records(cli.post("/api/getstatedistict/" + urllib.parse.quote(f"{state},{v}", safe=",")))
                except NgdrError as e:
                    note(f"getstatedistict/{state},{v}: {e}")
                    got = []
                if got:
                    used = v
                    break
            (out / "raw" / f"district_{state.replace(' ', '_')}_{dname.replace(' ', '_')}.json").write_text(
                json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
            for r in got:
                key = json.dumps(r, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    drows.append({"query_state": state, "query_district": used or dname, **r})
            print(f"  {state.title()} / {dname}: {len(got)} records" + ("" if used else " (no match under any spelling)"))
    write_csv(out / "district_commodities.csv", drows)
    coms = sorted({c.strip() for r in drows for c in str(r.get("commodity", "")).split(",") if c.strip()})
    note(f"District lists: {len(drows)} distinct records; commodities named: {', '.join(coms[:40])}"
         + (" ..." if len(coms) > 40 else ""))

    # 3. geology layers -> WMS
    try:
        gl = as_records(cli.post("/api/geologylayer"))
    except NgdrError as e:
        gl = []
        note(f"geologylayer: {e}")
    (out / "raw" / "geologylayer.json").write_text(json.dumps(gl, ensure_ascii=False, indent=1), encoding="utf-8")
    services = {}
    for e in gl:
        url = str(e.get("url", "")).strip()
        names = [n.strip() for n in str(e.get("layername", "")).split(",") if n.strip()]
        if url:
            services.setdefault(url, []).extend(names)
    if not services:
        note("geologylayer returned no WMS address; skipping maps.")
    peaks = prospects_table(proc, args.model)
    fi_rows, layer_rows = [], []
    for url, documented in services.items():
        note(f"WMS {url}: documented layers {', '.join(documented)}")
        caps_layers, map_fmts, fi_fmts, version = [], [], [], "1.1.1"
        try:
            raw, _ = cli.get(wms_url(url, SERVICE="WMS", REQUEST="GetCapabilities", VERSION="1.1.1"))
            (out / "raw" / "wms_capabilities.xml").write_bytes(raw)
            caps_layers, map_fmts, fi_fmts, version = parse_wms_caps(raw)
            version = version or "1.1.1"
            note(f"  GetCapabilities: {len(caps_layers)} layers published (WMS {version})")
        except Exception as ex:
            note(f"  GetCapabilities failed ({type(ex).__name__}); using the documented layer names only")
        layer_rows += [{"service": url, **lyr} for lyr in caps_layers]
        by_name = {lyr["name"]: lyr for lyr in caps_layers}
        wanted = list(dict.fromkeys(documented + [lyr["name"] for lyr in caps_layers
                                                  if args.all_layers or LAYER_KEYWORDS.search(lyr["name"] + " " + lyr["title"])]))
        fmt = "image/png" if not map_fmts or "image/png" in map_fmts else map_fmts[0]
        info_fmt = next((f for f in ("application/json", "text/plain", "application/vnd.ogc.gml", "text/html")
                         if not fi_fmts or f in fi_fmts), "text/plain")
        w = args.map_px
        h = int(round(w * (AOI[3] - AOI[1]) / (AOI[2] - AOI[0])))
        for name in wanted:
            safe = re.sub(r"[^\w.-]", "_", name)
            params = dict(SERVICE="WMS", REQUEST="GetMap", VERSION="1.1.1", LAYERS=name, STYLES="",
                          SRS="EPSG:4326", BBOX=",".join(map(str, AOI)), WIDTH=w, HEIGHT=h, FORMAT=fmt, TRANSPARENT="true")
            try:
                raw, ctype = cli.get(wms_url(url, **params))
                err = is_xml_error(raw, ctype)
                if err:
                    note(f"  map {name}: server error: {err}")
                else:
                    (out / "wms_maps" / f"{safe}.png").write_bytes(raw)
                    ok = png_to_geotiff(raw, out / "wms_maps" / f"{safe}.tif", AOI, w, h)
                    note(f"  map {name}: saved{' (georeferenced .tif)' if ok else ''}")
            except Exception as ex:
                note(f"  map {name}: failed ({type(ex).__name__}: {str(ex)[:80]})")
            # feature info at prospect peaks
            if peaks and (not by_name or by_name.get(name, {}).get("queryable", True)):
                n_ok = 0
                for p in peaks:
                    d = 0.0005
                    q = dict(SERVICE="WMS", REQUEST="GetFeatureInfo", VERSION="1.1.1", LAYERS=name, QUERY_LAYERS=name,
                             STYLES="", SRS="EPSG:4326", BBOX=f"{p['lon'] - d},{p['lat'] - d},{p['lon'] + d},{p['lat'] + d}",
                             WIDTH=101, HEIGHT=101, X=50, Y=50, INFO_FORMAT=info_fmt, FEATURE_COUNT=5, FORMAT=fmt)
                    row = {"prospect": p["prospect"], "tier": p["tier"], "lon": p["lon"], "lat": p["lat"], "layer": name}
                    try:
                        raw, ctype = cli.get(wms_url(url, **q), timeout=60)
                        err = is_xml_error(raw, ctype)
                        if err:
                            row["error"] = err
                        elif "json" in ctype or raw.lstrip()[:1] in (b"{", b"["):
                            js = json.loads(raw)
                            feats = js.get("features", []) if isinstance(js, dict) else []
                            row["n_features"] = len(feats)
                            row["properties"] = [f.get("properties", {}) for f in feats]
                            n_ok += bool(feats)
                        else:
                            txt = re.sub(r"<[^>]+>", " ", raw.decode("utf-8", "replace"))
                            row["text"] = re.sub(r"\s+", " ", txt).strip()[:500]
                            n_ok += bool(row["text"])
                    except Exception as ex:
                        row["error"] = f"{type(ex).__name__}"
                    fi_rows.append(row)
                print(f"    feature info {name}: {n_ok}/{len(peaks)} prospects returned something")

        # WFS: only if the same address also serves it
        try:
            raw, ctype = cli.get(wms_url(url, SERVICE="WFS", REQUEST="GetCapabilities", VERSION="1.0.0"))
            root = ET.fromstring(raw)
            if local(root.tag) != "WFS_Capabilities":
                raise ValueError("not a WFS reply")
            (out / "raw" / "wfs_capabilities.xml").write_bytes(raw)
            ftypes = [child_text(ft, "Name") for ft in find_all(root, "FeatureType")]
            note(f"  WFS offered: {len(ftypes)} feature types")
            short = {d.split(":")[-1] for d in documented}
            for name in [t for t in ftypes if t.split(":")[-1] in short]:   # the documented geology layers only
                q = dict(SERVICE="WFS", REQUEST="GetFeature", VERSION="1.0.0", TYPENAME=name, SRSNAME="EPSG:4326",
                         BBOX=",".join(map(str, AOI)), OUTPUTFORMAT="application/json", MAXFEATURES=50000)
                try:
                    raw, ctype = cli.get(wms_url(url, **q), timeout=300)
                    js = json.loads(raw)
                    path = out / "wfs" / (re.sub(r"[^\w.-]", "_", name) + ".geojson")
                    path.write_text(json.dumps(js), encoding="utf-8")
                    note(f"    {name}: {len(js.get('features', []))} features -> {path.name}")
                except Exception as ex:
                    note(f"    {name}: GetFeature failed ({type(ex).__name__})")
        except Exception:
            note("  WFS not offered at this address (WMS pictures only).")

    write_csv(out / "wms_layers.csv", layer_rows)
    write_csv(out / "wms_featureinfo.csv", fi_rows)
    report += ["", "What these are good for:",
               "  commodity_*.csv / district_commodities.csv: which commodities NGDR holds exploration records for in",
               "    these districts. Records name states and districts only (no coordinates), so they cannot become labels.",
               "  wms_maps/*.tif: 1:2M geology pictures for visual overlay in QGIS. Coarser than the 1:50K/250K maps",
               "    already in the model; not used as features.",
               "  wms_featureinfo.csv: 1:2M map attributes at each prospect peak, a cross-check of lithology.",
               "  The exploration reports themselves are not in this API: download them on the NGDR portal",
               "    (models/<model>/step4/report_pull_list.csv) and record results in data/processed/report_findings.csv.",
               "Credit: Geological Survey of India, National Geoscience Data Repository (NGDR)."]
    (out / "ngdr_pull_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
