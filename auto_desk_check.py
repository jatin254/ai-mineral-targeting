#!/usr/bin/env python3
"""
auto_desk_check.py - automated desk check of the tier A/B targets (replaces the manual Google Earth pass).

Run from the repository folder after run_v1.bat. Needs internet; first run takes about
5-15 minutes, reruns are fast because downloads are cached:
    python auto_desk_check.py data/processed data/raw/aikosh-gsi-ka-ap --model pg_ens
After editing check_status by hand in desk_check_auto.csv, re-tier without downloading anything:
    python auto_desk_check.py data/processed data/raw/aikosh-gsi-ka-ap --model pg_ens --apply-only

What it does
 1. Runs step 0 (prepare_step4.py): target list and GSI report pull list.
 2. For every target in a tier A/B prospect, measures from open data:
      ESA WorldCover 2021, 10 m            water, built-up, cropland, trees, bare/shrub/grass
      JRC Global Surface Water 1984-2021   how often the ground has been under water (seasonal tanks)
      Copernicus DEM GLO-30                relief and slope (ridges and hills vs flat plains)
      Sentinel-2 L2A (Earth Search)        dry-season greenness, post-monsoon water, true-colour images
      OpenStreetMap (Overpass API)         quarries, mine shafts, settlements, nearest road or track
      GSI 250K geomorphology (local)       flood plain, sand sheet, quarry, mine dump
 3. Assigns the manual check codes (keep, outcrop, old_workings, quarry, drop_water, drop_urban,
    drop_alluvium, unclear) with a confidence and the reasons, and marks rows that need a human look.
 4. Re-tiers the prospects: drop_* targets are removed, quarry targets move one tier down,
    old_workings targets become tier A.
Writes models/<model>/step4/
   desk_check_auto.csv        the sheet with check_status filled in; needs_eye = worth a look
   desk_check_report.html     open in a browser: images and evidence per target, needs_eye first
   prospects_after_check.csv  prospects re-tiered
   cache/                     downloads (delete the folder to force fresh data)
A source that cannot be reached is skipped and reported; the check then rests on the others.

Data credits (cite in anything you publish): ESA WorldCover 2021 (CC BY 4.0); EC JRC / Google Global
Surface Water v1.4; Copernicus DEM GLO-30 and Copernicus Sentinel-2 data (ESA), via Element 84 Earth
Search; OpenStreetMap contributors (ODbL); GSI geomorphology. These are screening layers: a target the
script keeps still needs ground truth before any field spend.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# GDAL network settings must be in place before rasterio loads GDAL
for k, v in {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2", "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "25000000",
}.items():
    os.environ.setdefault(k, v)
try:  # conda GDAL on Windows sometimes cannot find a CA bundle for https
    import certifi
    os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
try:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.merge import merge
    from rasterio.transform import from_bounds as transform_from_bounds
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds
    from shapely.geometry import LineString, Point, Polygon, box, mapping
    from shapely.ops import transform as shp_transform
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency: {e.name}. Run: conda env update -n te -f environment.yml")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_step4  # noqa: E402

# ----------------------------------------------------------------------------- sources (overridable)
WORLDCOVER_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
GSW_URL = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif"
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
STAC_COLLECTIONS = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]
USER_AGENT = "ai-mineral-targeting-desk-check/1.0"

DRY_WINDOWS = [("2026-02-01", "2026-04-30"), ("2025-02-01", "2025-04-30"), ("2024-02-01", "2024-04-30")]
WET_WINDOWS = [("2025-10-01", "2025-12-20"), ("2024-10-01", "2024-12-20"), ("2026-10-01", "2026-12-20")]
DRY_MAX_CLOUD, WET_MAX_CLOUD = 10, 25

# ----------------------------------------------------------------------------- decision thresholds
WATER_DROP = 0.25          # share of the target under water (any source) that drops it
WATER_DROP_LOWRAD = 0.10   # stricter for targets flagged low_radiometric
BUILT_DROP = 0.25
NATURAL_OUTCROP = 0.40     # bare + shrub + grass share
RELIEF_OUTCROP_M = 25.0    # p95 - p5 elevation inside the target
SLOPE_OUTCROP_DEG = 4.0
CROP_FLAT = 0.75         # cropland share that, on flat ground, means soil-covered (kept, low confidence)
RELIEF_FLAT_M, SLOPE_FLAT_DEG = 10.0, 1.5
OSM_MARGIN_M = 1000.0      # OpenStreetMap query box: prospect extent plus this margin
QUARRY_SHARE = 0.10       # quarry covering this share of the target moves it a tier down
QUARRY_PEAK_M = 200.0      # ... as does a quarry this close to the target's peak cell
MIN_CLEAR = 0.6            # Sentinel-2 metrics need this share of cloud-free pixels

WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow",
              80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
STATUS_COLOUR = {"keep": "#4c9a2a", "outcrop": "#1b7837", "old_workings": "#b8860b", "quarry": "#d95f02",
                 "drop_water": "#2166ac", "drop_urban": "#762a83", "drop_alluvium": "#8c6d31", "unclear": "#777777"}
DROP_CODES = {"drop_water", "drop_urban", "drop_alluvium"}   # target removed
DEMOTE = {"A": "B", "B": "C", "C": "C"}                     # quarry: one tier down

TO_LL = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


# ----------------------------------------------------------------------------- helpers
class Cache:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def key(self, *parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def get(self, kind: str, key: str):
        p = self.root / kind / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, kind: str, key: str, value) -> None:
        (self.root / kind).mkdir(parents=True, exist_ok=True)
        (self.root / kind / f"{key}.json").write_text(json.dumps(value, default=float))

    def chip_path(self, key: str) -> Path:
        (self.root / "chips").mkdir(parents=True, exist_ok=True)
        return self.root / "chips" / f"{key}.jpg"


def http(url: str, data: bytes | None = None, timeout: int = 60, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def reachable(url: str) -> bool:
    """Cheap check before the slow GDAL path: a 1-byte range request."""
    if not url.startswith("http"):
        return Path(url).exists()
    try:
        http(url, timeout=15, headers={"Range": "bytes=0-0"})
        return True
    except urllib.error.HTTPError as e:
        return e.code in (200, 206, 416)
    except Exception:
        return False


def gdal_path(url: str) -> str:
    return "/vsicurl/" + url if url.startswith("http") else url


def ll(geom):
    return shp_transform(TO_LL.transform, geom)


def frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def pct(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(v) else f"{v:.0%}"


def fnum(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(v) else f"{v:.{nd}f}"


# ----------------------------------------------------------------------------- tiles
def worldcover_tiles(b):
    out = set()
    for lat in range(int(math.floor(b[1] / 3) * 3), int(math.floor(b[3] / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(b[0] / 3) * 3), int(math.floor(b[2] / 3) * 3) + 1, 3):
            out.add(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return sorted(out)


def gsw_tiles(b):
    out = set()
    for lon in range(int(math.floor(b[0] / 10) * 10), int(math.floor(b[2] / 10) * 10) + 1, 10):
        for top in range(int(math.ceil(b[1] / 10) * 10), int(math.ceil(b[3] / 10) * 10) + 1, 10):
            if top == b[1]:  # a box starting exactly on a tile edge belongs to the tile above
                top += 10
            out.add(f"{abs(lon)}{'E' if lon >= 0 else 'W'}_{abs(top)}{'N' if top >= 0 else 'S'}")
    return sorted(out)


def dem_tiles(b):
    out = set()
    for lat in range(int(math.floor(b[1])), int(math.floor(b[3])) + 1):
        for lon in range(int(math.floor(b[0])), int(math.floor(b[2])) + 1):
            out.add(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00")
    return sorted(out)


class TileReader:
    """Reads a lon/lat window from a set of tiles (local or remote), merging across tile edges."""

    def __init__(self, name, template, tiler, nodata):
        self.name, self.template, self.tiler, self.nodata = name, template, tiler, nodata
        self.ds, self.failed, self.ok = {}, set(), True

    def _open(self, url):
        if url in self.failed:
            return None
        if url not in self.ds:
            try:
                self.ds[url] = rasterio.open(gdal_path(url))
            except Exception:
                self.failed.add(url)
                return None
        return self.ds[url]

    def read(self, bounds):
        dss = [d for d in (self._open(self.template.format(tile=t)) for t in self.tiler(bounds)) if d is not None]
        if not dss:
            raise RuntimeError(f"{self.name}: no tile could be opened")
        arr, tr = merge(dss, bounds=bounds, nodata=self.nodata)
        return arr[0], tr


# ----------------------------------------------------------------------------- per-target measurements
def landcover_metrics(reader: TileReader, g_ll) -> dict:
    b = g_ll.bounds
    arr, tr = reader.read((b[0] - 0.001, b[1] - 0.001, b[2] + 0.001, b[3] + 0.001))
    inside = geometry_mask([mapping(g_ll)], out_shape=arr.shape, transform=tr, invert=True)
    if inside.sum() < 20:
        inside = geometry_mask([mapping(g_ll)], out_shape=arr.shape, transform=tr, invert=True, all_touched=True)
    v = arr[inside]
    v = v[v != 0]
    if v.size == 0:
        raise RuntimeError("no land-cover pixels")
    out = {f"wc_{n}": frac(v == c) for c, n in WC_CLASSES.items() if c in (10, 20, 30, 40, 50, 60, 80, 90)}
    out["wc_natural"] = out["wc_bare"] + out["wc_shrub"] + out["wc_grass"]
    out["wc_water_all"] = out["wc_water"] + out["wc_wetland"]
    return out


def gsw_metrics(reader: TileReader, g_ll) -> dict:
    b = g_ll.bounds
    arr, tr = reader.read((b[0] - 0.001, b[1] - 0.001, b[2] + 0.001, b[3] + 0.001))
    inside = geometry_mask([mapping(g_ll)], out_shape=arr.shape, transform=tr, invert=True, all_touched=True)
    v = arr[inside].astype("int16")
    v = v[v <= 100]   # 255 = no data
    if v.size == 0:
        raise RuntimeError("no surface-water pixels")
    return {"gsw_ever": frac(v > 0), "gsw_seasonal": frac(v >= 10), "gsw_frequent": frac(v >= 50),
            "gsw_max_occ": float(v.max())}


def dem_metrics(reader: TileReader, g_ll) -> dict:
    b = g_ll.bounds
    arr, tr = reader.read((b[0] - 0.003, b[1] - 0.003, b[2] + 0.003, b[3] + 0.003))
    z = arr.astype("float64")
    z[z <= -1000] = np.nan
    lat = (b[1] + b[3]) / 2
    dx = abs(tr.a) * 111320 * math.cos(math.radians(lat))
    dy = abs(tr.e) * 110574
    gy, gx = np.gradient(np.where(np.isfinite(z), z, np.nanmean(z)), dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    inside = geometry_mask([mapping(g_ll)], out_shape=z.shape, transform=tr, invert=True, all_touched=True)
    zi = z[inside & np.isfinite(z)]
    if zi.size < 4:
        raise RuntimeError("no elevation pixels")
    return {"dem_relief_m": float(np.percentile(zi, 95) - np.percentile(zi, 5)),
            "dem_slope_deg": float(np.mean(slope[inside])), "dem_elev_m": float(np.median(zi))}


GM_WATER = "water|reservoir|tank|pond|lake|dam|river"
GM_EMPTY = {"gm_classes": "", "gm_floodplain_sand": 0.0, "gm_water_frac": 0.0, "gm_quarry_frac": 0.0,
            "gm_quarry_peak_m": float("inf"), "gm_old_dump": False}


def geomorph_metrics(gm: gpd.GeoDataFrame | None, geom, peak=None) -> dict:
    if gm is None:
        return dict(GM_EMPTY)
    near = gm[gm.intersects(geom.buffer(250))]
    hit = near[near.intersects(geom)]
    if hit.empty:
        return dict(GM_EMPTY)
    area = hit.geometry.intersection(geom).area / geom.area
    lab = hit["_label"].astype(str)
    low = lab.str.lower()
    parts = sorted(zip(area, lab), reverse=True)
    quarry_all = near[near["_label"].astype(str).str.lower().str.contains("active quarry")]
    qpeak = min((peak.distance(q) for q in quarry_all.geometry), default=float("inf")) if peak is not None else float("inf")
    return {"gm_classes": "; ".join(f"{l} {a:.0%}" for a, l in parts[:3]),
            "gm_floodplain_sand": float(area[low.str.contains("flood plain|aeolian|sand sheet")].sum()),
            "gm_water_frac": float(min(area[low.str.contains(GM_WATER)].sum(), 1.0)),
            "gm_quarry_frac": float(area[low.str.contains("active quarry")].sum()),
            "gm_quarry_peak_m": float(qpeak),
            "gm_old_dump": bool(low.str.contains("mine dump|abandoned quarry").any())}


# ----------------------------------------------------------------------------- OpenStreetMap
def overpass_query(s, w, n, e) -> dict:
    q = f"""[out:json][timeout:50];
(
  way["landuse"~"^(quarry|industrial|residential|landfill)$"]({s},{w},{n},{e});
  way["man_made"~"^(mineshaft|adit|mine)$"]({s},{w},{n},{e});
  node["man_made"~"^(mineshaft|adit)$"]({s},{w},{n},{e});
  node["historic"~"^(mine|mine_shaft|mine_adit)$"]({s},{w},{n},{e});
  way["historic"~"^(mine|mine_shaft)$"]({s},{w},{n},{e});
  way["highway"]({s},{w},{n},{e});
);
out geom;"""
    last = None
    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                raw = http(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=65)
                return json.loads(raw)
            except urllib.error.HTTPError as ex:
                last = ex
                if ex.code in (429, 502, 503, 504):   # busy or rate-limited: wait, then retry
                    time.sleep(10 * (attempt + 1))
                    continue
                break
            except Exception as ex:                  # no connection or bad reply: next mirror
                last = ex
                break
    raise RuntimeError(f"Overpass unavailable ({type(last).__name__}: {str(last)[:60]})")


def osm_features(js: dict) -> dict:
    quarry, settle, mines, roads = [], [], [], []
    for el in js.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            mines.append(Point(TO_UTM.transform(el["lon"], el["lat"])))
            continue
        pts = [TO_UTM.transform(p["lon"], p["lat"]) for p in el.get("geometry", []) if p]
        if len(pts) < 2:
            continue
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        if "highway" in tags:
            roads.append(LineString(pts))
        elif tags.get("landuse") == "quarry" and closed:
            quarry.append(Polygon(pts).buffer(0))
        elif tags.get("landuse") in ("industrial", "residential", "landfill") and closed:
            settle.append(Polygon(pts).buffer(0))
        elif tags.get("man_made") or tags.get("historic"):
            mines.append(Polygon(pts).buffer(0) if closed else LineString(pts))
    return {"quarry": quarry, "settle": settle, "mines": mines, "roads": roads}


def osm_metrics(f: dict, geom, peak) -> dict:
    a = geom.area
    q = sum(geom.intersection(p).area for p in f["quarry"]) / a if f["quarry"] else 0.0
    qd = min((geom.distance(p) for p in f["quarry"]), default=float("inf"))
    qp = min((peak.distance(p) for p in f["quarry"]), default=float("inf"))
    st = sum(geom.intersection(p).area for p in f["settle"]) / a if f["settle"] else 0.0
    zone = geom.buffer(200)
    mines = sum(1 for m in f["mines"] if zone.intersects(m))
    rd = min((peak.distance(r) for r in f["roads"]), default=float("inf"))
    return {"osm_quarry_frac": float(q), "osm_quarry_dist_m": float(qd), "osm_quarry_peak_m": float(qp),
            "osm_settlement_frac": float(min(st, 1.0)),
            "osm_mine_features": int(mines), "road_dist_m": float(rd)}


# ----------------------------------------------------------------------------- Sentinel-2
def stac_search(lon: float, lat: float, start: str, end: str, max_cloud: int) -> list[dict]:
    feats = []
    for coll in STAC_COLLECTIONS:
        body = {"collections": [coll], "intersects": {"type": "Point", "coordinates": [lon, lat]},
                "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 10,
                "query": {"eo:cloud_cover": {"lt": max_cloud}}}
        try:
            js = json.loads(http(STAC_URL, data=json.dumps(body).encode(), timeout=60,
                                 headers={"Content-Type": "application/json"}))
            feats = js.get("features", [])
        except Exception:
            feats = []
        if feats:
            break
    return sorted(feats, key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100))


S2_ASSET_NAMES = {"red": ("red", "B04"), "green": ("green", "B03"), "blue": ("blue", "B02"), "nir": ("nir", "B08"),
                  "swir16": ("swir16", "B11"), "scl": ("scl", "SCL"), "visual": ("visual", "TCI")}


def s2_assets(assets: dict) -> dict:
    """Earth Search names assets by colour; some catalogues use band ids. Accept either."""
    out = {}
    for key, names in S2_ASSET_NAMES.items():
        for nm in names:
            if nm in assets and "href" in assets[nm]:
                out[key] = {"href": assets[nm]["href"], "bands": assets[nm].get("raster:bands")}
                break
    return out


def pick_scenes(lon, lat, windows, max_cloud, cache: Cache, n_max: int = 4) -> list[dict]:
    """Up to n_max candidate scenes, most recent window first, least cloudy first within a window."""
    key = cache.key("stac2", round(lon, 2), round(lat, 2), windows, max_cloud, n_max)
    hit = cache.get("stac", key)
    if hit is not None:
        return hit
    keep = []
    for start, end in windows:
        for f in stac_search(lon, lat, start, end, max_cloud):
            keep.append({"id": f.get("id"), "datetime": f["properties"].get("datetime", "")[:10],
                         "cloud": f["properties"].get("eo:cloud_cover"),
                         "boa_offset_applied": f["properties"].get("earthsearch:boa_offset_applied"),
                         "baseline": f["properties"].get("s2:processing_baseline"),
                         "assets": s2_assets(f.get("assets", {}))})
            if len(keep) >= n_max:
                break
        if len(keep) >= n_max:
            break
    cache.put("stac", key, keep)
    return keep


def scale_offset(scene: dict, asset: str) -> tuple[float, float]:
    bands = (scene["assets"].get(asset) or {}).get("bands") or [{}]
    b0 = bands[0] if bands else {}
    if "scale" in b0 or "offset" in b0:
        return float(b0.get("scale", 1e-4)), float(b0.get("offset", 0.0))
    try:
        newer = float(scene.get("baseline") or 0) >= 4.0
    except ValueError:
        newer = True
    return 1e-4, (-0.1 if newer and not scene.get("boa_offset_applied") else 0.0)


def read_band(href, bounds_utm, shape_hw, resampling):
    with rasterio.open(gdal_path(href)) as ds:
        b = transform_bounds("EPSG:32643", ds.crs, *bounds_utm) if str(ds.crs) != "EPSG:32643" else bounds_utm
        win = window_from_bounds(*b, transform=ds.transform)
        return ds.read(window=win, out_shape=(ds.count, *shape_hw), boundless=True, fill_value=0,
                       resampling=resampling)


def reflectance(scene, asset, bounds, n):
    so = scale_offset(scene, asset)
    raw = read_band(scene["assets"][asset]["href"], bounds, (n, n), Resampling.bilinear)[0].astype("float64")
    return raw * so[0] + so[1]


def s2_for_target(t: dict, cache: Cache) -> dict:
    """Dry-season NDVI, post-monsoon water share, and two true-colour chips."""
    geom = t["geometry"]
    cx, cy = geom.centroid.x, geom.centroid.y
    side = max(1600.0, max(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1]) + 600)
    bounds = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    n = int(min(260, side / 10))
    tr = transform_from_bounds(*bounds, n, n)
    inside = geometry_mask([mapping(geom)], out_shape=(n, n), transform=tr, invert=True, all_touched=True)
    lon, lat = TO_LL.transform(cx, cy)
    out = {}
    for season, windows, maxc in (("dry", DRY_WINDOWS, DRY_MAX_CLOUD), ("wet", WET_WINDOWS, WET_MAX_CLOUD)):
        key = cache.key("s2v3", t["target"], [round(v) for v in bounds], season)
        hit = cache.get("s2", key)
        if hit is not None:
            out.update(hit)
            continue
        res = {f"s2_{season}_date": "", f"s2_{season}_chip": ""}
        try:
            # the tile-wide cloud figure says little about one small target: take the first scene
            # that is clear over the target, else the clearest one tried
            best = None
            for scene in pick_scenes(lon, lat, windows, maxc, cache):
                a = scene["assets"]
                if "scl" in a:
                    scl = read_band(a["scl"]["href"], bounds, (n, n), Resampling.nearest)[0]
                    clear = inside & (scl > 0) & ~np.isin(scl, [3, 8, 9, 10])
                else:
                    scl, clear = None, inside.copy()
                cf = clear.sum() / max(inside.sum(), 1)
                if best is None or cf > best[3]:
                    best = (scene, scl, clear, cf)
                if cf >= MIN_CLEAR:
                    break
            if best is not None:
                scene, scl, clear, cf = best
                a = scene["assets"]
                res.update({f"s2_{season}_date": scene["datetime"], f"s2_{season}_clear": float(cf)})
                if season == "dry" and {"red", "nir"} <= set(a) and cf >= MIN_CLEAR:
                    red, nir = reflectance(scene, "red", bounds, n), reflectance(scene, "nir", bounds, n)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        ndvi = (nir - red) / (nir + red)
                    res["s2_dry_ndvi"] = float(np.nanmean(ndvi[clear]))
                if season == "wet" and {"green", "swir16"} <= set(a) and cf >= MIN_CLEAR:
                    g, s = reflectance(scene, "green", bounds, n), reflectance(scene, "swir16", bounds, n)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        water = (g - s) / (g + s) > 0
                    if scl is not None:
                        water |= scl == 6
                    res["s2_wet_water"] = float(water[clear].mean())
                rgb = None
                if "visual" in a:
                    rgb = read_band(a["visual"]["href"], bounds, (n, n), Resampling.bilinear)[:3]
                elif {"red", "green", "blue"} <= set(a):
                    rgb = np.stack([reflectance(scene, k, bounds, n) for k in ("red", "green", "blue")])
                if rgb is not None:
                    path = cache.chip_path(key)
                    cap = scene["datetime"] + (f"  {cf:.0%} clear" if "scl" in a else "")
                    render_chip(rgb, geom, bounds, path, cap)
                    res[f"s2_{season}_chip"] = str(path)
        except Exception as ex:
            res[f"s2_{season}_error"] = f"{type(ex).__name__}: {str(ex)[:80]}"
        cache.put("s2", key, res)
        out.update(res)
    return out


def render_chip(rgb, geom, bounds, path: Path, caption: str) -> None:
    """Thread-safe (no pyplot). Percentile stretch so bright, dry ground stays readable."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    img = np.transpose(np.asarray(rgb, dtype="float64"), (1, 2, 0))
    valid = img.sum(axis=2) > 0
    if valid.any():
        lo, hi = np.percentile(img[valid], [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    img[~valid] = 0.85
    fig = Figure(figsize=(2.6, 2.75), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0.07, 1, 0.93])
    ax.imshow(img, extent=(bounds[0], bounds[2], bounds[1], bounds[3]), interpolation="bilinear")
    for p in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
        x, y = p.exterior.xy
        ax.plot(x, y, color="#ffde00", lw=1.3)
    ax.plot([bounds[2] - 650, bounds[2] - 150], [bounds[1] + 120] * 2, color="white", lw=2)
    ax.text(bounds[2] - 400, bounds[1] + 170, "500 m", color="white", ha="center", fontsize=6)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_axis_off()
    fig.text(0.5, 0.015, caption, ha="center", fontsize=7)
    fig.savefig(path, format="jpeg", pil_kwargs={"quality": 82})


# ----------------------------------------------------------------------------- decision
def decide(m: dict, flags: str) -> tuple[str, str, list[str], bool]:
    fl = {f.strip() for f in str(flags or "").split(",") if f.strip() and f.strip() != "nan"}

    def g(k):  # number or None (missing, blank, NaN, error text)
        try:
            v = float(m.get(k))
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    def b(k):  # True only for a real True (NaN from a missing row is truthy in Python)
        v = m.get(k)
        return isinstance(v, (bool, np.bool_)) and bool(v) or (isinstance(v, str) and v.strip().lower() == "true")

    water_names = {"wc_water_all": "WorldCover", "gsw_seasonal": "surface-water history",
                   "s2_wet_water": "post-monsoon image", "gm_water_frac": "GSI landform map"}
    water_srcs = {k: g(k) for k in water_names if g(k) is not None}
    water = max(water_srcs.values(), default=None)
    built = max([v for v in (g("wc_built"), g("osm_settlement_frac")) if v is not None], default=None)
    relief, slope = g("dem_relief_m"), g("dem_slope_deg")
    natural, crop, ndvi = g("wc_natural"), g("wc_crop"), g("s2_dry_ndvi")
    have_cover = g("wc_crop") is not None
    eye = bool(fl & {"felsic_radiometric", "ratio_extreme"})

    # quarries: a small one at the edge is a note, not a verdict
    q_share = max(g("osm_quarry_frac") or 0.0, g("gm_quarry_frac") or 0.0)
    q_peak = min(v for v in (g("osm_quarry_peak_m"), g("gm_quarry_peak_m"), float("inf")) if v is not None)
    q_any = q_share > 0 or (g("osm_quarry_dist_m") or float("inf")) <= 150 or q_peak <= 500
    notes = [f"quarry nearby ({q_share:.0%} of target, {q_peak:.0f} m from peak): check the image"] \
        if q_any and not (q_share >= QUARRY_SHARE or q_peak <= QUARRY_PEAK_M) else []

    def done(code, conf, reasons, look):
        return code, conf, reasons + notes, bool(look or notes)

    if not have_cover and water is None:
        return done("unclear", "low", ["no land-cover or water data could be read"], True)
    thr = WATER_DROP_LOWRAD if "low_radiometric" in fl else WATER_DROP
    if water is not None and water >= thr:
        agree = sum(v >= thr for v in water_srcs.values())
        sure = agree >= 2 or water >= 2 * thr
        reasons = ["water: " + ", ".join(f"{water_names[k]} {v:.0%}" for k, v in water_srcs.items() if v > 0)]
        # a target only partly over water may still have real ground in it: ask for a look
        return done("drop_water", "high" if sure else "medium", reasons, eye or (not sure and water < 0.5))
    if built is not None and built >= BUILT_DROP:
        return done("drop_urban", "high" if built >= 0.4 else "medium", [f"built-up {built:.0%}"], eye)
    mines = g("osm_mine_features") or 0
    if mines > 0 or b("gm_old_dump"):
        src = "OSM mine/shaft" if mines > 0 else "GSI mine dump / abandoned quarry"
        return done("old_workings", "medium", [f"{src} on or within 200 m of the target"], True)
    if q_share >= QUARRY_SHARE or q_peak <= QUARRY_PEAK_M:
        why = f"quarry covers {q_share:.0%} of the target" if q_share >= QUARRY_SHARE else f"quarry {q_peak:.0f} m from the peak"
        return done("quarry", "medium", [why + ": stone quarry (artefact) or old workings? check the image"], True)
    if (g("gm_floodplain_sand") or 0) >= 0.3:
        return done("drop_alluvium", "medium", [f"GSI flood plain / sand sheet {g('gm_floodplain_sand'):.0%}"], eye)
    flat = relief is not None and slope is not None and relief < RELIEF_FLAT_M and slope < SLOPE_FLAT_DEG
    if crop is not None and crop >= CROP_FLAT and flat:
        # schist belts often weather to flat, farmed ground with thin soil, so this is not a drop:
        # imagery cannot tell soil thickness. Keep, but sampling there means soil, not rock chips.
        r = [f"soil-covered flat farmland (cropland {crop:.0%}, relief {relief:.0f} m)",
             "no outcrop: plan soil sampling"]
        if ndvi is not None:
            r.append(f"dry-season NDVI {ndvi:.2f}")
        return done("keep", "low", r, True)
    hilly = (relief is not None and relief >= RELIEF_OUTCROP_M) or (slope is not None and slope >= SLOPE_OUTCROP_DEG)
    if natural is not None and natural >= NATURAL_OUTCROP and hilly:
        # WorldCover cannot see rock: this is scrub/bare hill ground, where outcrop is likely
        conf = "high" if natural >= 0.6 and relief is not None and relief >= 40 else "medium"
        r = [f"scrub/bare hill ground {natural:.0%}", f"relief {relief:.0f} m" if relief is not None else f"slope {slope:.1f}°"]
        return done("outcrop", conf, r, eye)
    r = []
    if water is not None and water >= 0.10:
        r.append(f"partly water ({water:.0%})")
    if crop is not None:
        r.append(f"cropland {crop:.0%}")
    if natural is not None:
        r.append(f"scrub/bare {natural:.0%}")
    if relief is not None:
        r.append(f"relief {relief:.0f} m")
    if "felsic_radiometric" in fl:
        r.append("high K/Th: check for granite tors")
    conf = "medium" if have_cover and relief is not None else "low"
    return done("keep", conf, r, eye or conf == "low")


# ----------------------------------------------------------------------------- re-tier
def retier(sheet: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """drop_* removes a target, quarry moves it one tier down, old_workings moves it to tier A;
    a prospect takes the best tier of its remaining targets."""
    rank = {"A": 0, "B": 1, "C": 2}
    rows = []
    for p, g in sheet.groupby("prospect", sort=False):
        before = g["best_tier"].iloc[0]
        st = g["check_status"].fillna("").astype(str).str.strip()
        alive = g[~st.isin(DROP_CODES)].copy()
        ast = st[~st.isin(DROP_CODES)]
        alive.loc[ast == "quarry", "tier"] = alive.loc[ast == "quarry", "tier"].map(DEMOTE)
        alive.loc[ast == "old_workings", "tier"] = "A"
        after = min(alive["tier"], key=lambda x: rank.get(x, 9)) if len(alive) else "dropped"
        best = alive.sort_values("mean_pct", ascending=False).iloc[0] if len(alive) else g.iloc[0]
        road = pd.to_numeric(g["road_dist_m"], errors="coerce").min() if "road_dist_m" in g else float("nan")
        rows.append({"prospect": p, "tier_before": before, "tier_after": after, "n_targets": len(g),
                     "n_dropped": int(st.isin(DROP_CODES).sum()),
                     "statuses": " ".join(f"{k}:{v}" for k, v in st.value_counts().items()),
                     "best_target": best["target"], "peak_lat": best["peak_lat"], "peak_lon": best["peak_lon"],
                     "dist_known_gold_km": g["dist_known_gold_km"].min(),
                     "road_dist_m": round(road) if math.isfinite(road) else None,
                     "satellite_link": prepare_step4.maps_link(best["peak_lat"], best["peak_lon"])})
    out = pd.DataFrame(rows)
    if "gsi_gold_reports" in prospects:
        out = out.join(prospects.set_index("prospect")[["gsi_gold_reports"]], on="prospect")
    order = {"A": 0, "B": 1, "C": 2, "dropped": 3}
    return out.assign(_o=out["tier_after"].map(order)).sort_values(["_o", "prospect"]).drop(columns="_o")


# ----------------------------------------------------------------------------- report
def img_tag(path: str) -> str:
    if not path or not Path(path).exists():
        return '<div class="noimg">no clear image</div>'
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(Path(path).read_bytes()).decode()}" alt="">'


def write_html(sheet: pd.DataFrame, path: Path, sources: dict, after: pd.DataFrame) -> None:
    s = sheet.copy()
    s["_o"] = (~s["needs_eye"].astype(bool)).astype(int)
    s = s.sort_values(["_o", "order"])
    cards = []
    for _, r in s.iterrows():
        st = str(r["check_status"])
        metrics = [("water, any source", pct(r.get("water_max"))), ("built-up", pct(r.get("wc_built"))),
                   ("cropland", pct(r.get("wc_crop"))), ("bare / shrub / grass", pct(r.get("wc_natural"))),
                   ("trees", pct(r.get("wc_tree"))), ("ever under water 1984-2021", pct(r.get("gsw_ever"))),
                   ("relief (m)", fnum(r.get("dem_relief_m"), 0)), ("mean slope (°)", fnum(r.get("dem_slope_deg"), 1)),
                   ("dry-season NDVI", fnum(r.get("s2_dry_ndvi"))), ("peak to road/track (m)", fnum(r.get("road_dist_m"), 0))]
        mt = "".join(f"<tr><td>{html.escape(k)}</td><td>{v or '–'}</td></tr>" for k, v in metrics)
        link = (f"https://www.google.com/maps/@?api=1&map_action=map&center={r['peak_lat']:.5f},"
                f"{r['peak_lon']:.5f}&zoom=16&basemap=satellite")
        cards.append(f"""
<div class="card{' eye' if r['needs_eye'] else ''}">
  <div class="hd"><b>{html.escape(str(r['target']))}</b> · {html.escape(str(r['prospect']))} (tier {html.escape(str(r['best_tier']))})
    <span class="badge" style="background:{STATUS_COLOUR.get(st, '#555')}">{html.escape(st)}</span>
    <span class="conf">{html.escape(str(r['auto_confidence']))}</span>{' <span class="eyetag">look</span>' if r['needs_eye'] else ''}</div>
  <div class="imgs"><figure>{img_tag(r.get('s2_dry_chip', ''))}<figcaption>dry season</figcaption></figure>
    <figure>{img_tag(r.get('s2_wet_chip', ''))}<figcaption>post-monsoon</figcaption></figure></div>
  <div class="why">{html.escape(str(r['auto_reasons']))}</div>
  <div class="meta">status {html.escape(str(r['status']))} · {fnum(r['dist_known_gold_km'], 1)} km from known gold ·
    flags: {html.escape(str(r['flags']) if str(r['flags']) != 'nan' else 'none')}<br>
    evidence: {html.escape(str(r['evidence']))}<br>landform: {html.escape(str(r.get('gm_classes', '')))}</div>
  <table>{mt}</table>
  <a href="{link}" target="_blank" rel="noopener">open in Google Maps ↗</a>
</div>""")
    counts = " · ".join(f"{k} {v}" for k, v in sheet["check_status"].value_counts().items())
    tiers = " · ".join(f"{k} {v}" for k, v in after["tier_after"].value_counts().sort_index().items()) if len(after) else ""
    src = " · ".join(f"{k}: {'ok' if v else 'unavailable'}" for k, v in sources.items())
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Desk Check Report</title>
<style>
:root{{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--mute:#666;--line:#ddd;--eye:#e8a400}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--card:#222;--ink:#eee;--mute:#aaa;--line:#3a3a3a}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--mute);margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}}
.card.eye{{border:2px solid var(--eye)}} .hd{{margin-bottom:6px}}
.badge{{color:#fff;border-radius:4px;padding:1px 6px;font-size:12px;margin-left:4px}}
.conf{{color:var(--mute);font-size:12px;margin-left:4px}} .eyetag{{background:var(--eye);color:#000;border-radius:4px;padding:1px 5px;font-size:11px}}
.imgs{{display:flex;gap:6px}} figure{{margin:0;flex:1}} figure img{{width:100%;border-radius:4px;display:block}}
figcaption{{font-size:11px;color:var(--mute);text-align:center}}
.noimg{{aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:var(--line);border-radius:4px;color:var(--mute);font-size:12px}}
.why{{margin:6px 0;font-weight:600}} .meta{{color:var(--mute);font-size:12px;margin-bottom:6px}}
table{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:6px}} td{{border-top:1px solid var(--line);padding:2px 4px}}
td:last-child{{text-align:right}} a{{color:#2b6cb0}} footer{{margin-top:18px;color:var(--mute);font-size:12px}}
</style></head><body>
<h1>Automated desk check</h1>
<div class="sub">{len(sheet)} targets · {int(sheet['needs_eye'].sum())} marked <b>look</b> (shown first) ·
codes: {html.escape(counts)}<br>prospects after check: {html.escape(tiers)}<br>
Sources: {html.escape(src)}. Yellow outline = target. Override any check_status in desk_check_auto.csv and rerun with --apply-only.</div>
<div class="grid">{''.join(cards)}</div>
<footer>Contains modified Copernicus Sentinel-2 data (ESA) via Element 84 Earth Search; ESA WorldCover 2021 (CC BY 4.0);
EC JRC/Google Global Surface Water v1.4; Copernicus DEM GLO-30; © OpenStreetMap contributors (ODbL); GSI geomorphology.
Screening only: kept targets still need ground truth.</footer></body></html>"""
    path.write_text(doc, encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Automated desk check of tier A/B targets.")
    ap.add_argument("processed", help="data/processed")
    ap.add_argument("raw", nargs="?", default="", help="data/raw/aikosh-gsi-ka-ap (for the geomorphology map)")
    ap.add_argument("--model", default="pg_ens")
    ap.add_argument("--gpkg", default="ka_ap_v1.gpkg")
    ap.add_argument("--apply-only", action="store_true", help="re-tier from an edited desk_check_auto.csv, no downloads")
    ap.add_argument("--no-s2", action="store_true", help="skip Sentinel-2 (no images; faster)")
    ap.add_argument("--no-osm", action="store_true", help="skip OpenStreetMap")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    args = ap.parse_args()
    proc = Path(args.processed)
    mdir = proc / "models" / args.model
    out = mdir / "step4"
    t0 = time.time()

    if args.apply_only:
        sheet = pd.read_csv(out / "desk_check_auto.csv")
        prospects = pd.read_csv(mdir / "prospects.csv")
        after = retier(sheet, prospects)
        after.to_csv(out / "prospects_after_check.csv", index=False)
        write_html(sheet, out / "desk_check_report.html", {"your edits to desk_check_auto.csv": True}, after)
        print_summary(sheet, after, out, t0)
        return

    print("Step 0: preparing the target list and report pull list")
    out = prepare_step4.build(proc, args.raw, args.model, args.gpkg)
    sheet = pd.read_csv(out / "desk_check_sheet.csv")
    geo = gpd.read_file(out / "desk_check.gpkg", layer="targets_to_check").to_crs("EPSG:32643")
    geo = geo.set_index("target").geometry
    prospects = pd.read_csv(mdir / "prospects.csv")
    cache = Cache(out / "cache")

    gm = None
    gm_hits = sorted(Path(args.raw).glob("Geomorphology*/*.shp")) if args.raw else []
    if gm_hits:
        gm = gpd.read_file(gm_hits[0]).to_crs("EPSG:32643")
        cols = [c for c in ("descriptio", "legend_sho") if c in gm.columns] or [gm.columns[0]]
        gm["_label"] = gm[cols[0]].astype(str)
        gm = gm[gm.geometry.notna()]

    # which remote sources answer at all?
    aoi_ll = ll(box(*geo.total_bounds)).bounds
    sources = {
        "WorldCover": reachable(WORLDCOVER_URL.format(tile=worldcover_tiles(aoi_ll)[0])),
        "Surface Water": reachable(GSW_URL.format(tile=gsw_tiles(aoi_ll)[0])),
        "Copernicus DEM": reachable(DEM_URL.format(tile=dem_tiles(aoi_ll)[0])),
        "Sentinel-2": (not args.no_s2) and reachable(STAC_URL.replace("/search", "/collections")),
        "OpenStreetMap": (not args.no_osm),
        "Geomorphology": gm is not None,
    }
    print("Sources: " + ", ".join(f"{k} {'ok' if v else 'unavailable'}" for k, v in sources.items()))
    if not any(sources[k] for k in ("WorldCover", "Surface Water", "Copernicus DEM", "Sentinel-2")):
        print("  WARNING: none of the online image sources answered. Check the internet connection; behind an\n"
              "  office proxy, set HTTPS_PROXY first (set HTTPS_PROXY=http://host:port). Without them most\n"
              "  targets will come out 'unclear'.")

    use = {"wc": sources["WorldCover"], "gsw": sources["Surface Water"], "dem": sources["Copernicus DEM"]}
    fns = {"wc": landcover_metrics, "gsw": gsw_metrics, "dem": dem_metrics}
    local = threading.local()  # rasterio datasets must not be shared between threads
    peaks = {t: Point(TO_UTM.transform(lo, la)) for t, lo, la in zip(sheet["target"], sheet["peak_lon"], sheet["peak_lat"])}

    def measure(tid):
        geom = geo.get(tid)
        if geom is None:
            return tid, {}
        if not hasattr(local, "readers"):
            local.readers = {"wc": TileReader("WorldCover", WORLDCOVER_URL, worldcover_tiles, 0),
                             "gsw": TileReader("Surface Water", GSW_URL, gsw_tiles, 255),
                             "dem": TileReader("Copernicus DEM", DEM_URL, dem_tiles, -9999)}
        g_ll = ll(geom)
        rec = geomorph_metrics(gm, geom, peaks.get(tid))
        for k, fn in fns.items():
            if not use[k]:
                continue
            key = cache.key(k, tid, [round(v, 5) for v in g_ll.bounds])
            hit = cache.get(k, key)
            if hit is None:
                try:
                    hit = fn(local.readers[k], g_ll)
                    cache.put(k, key, hit)   # errors are not cached, so a rerun retries them
                except Exception as ex:
                    hit = {f"{k}_error": f"{type(ex).__name__}: {str(ex)[:80]}"}
            rec.update(hit)
        return tid, rec

    records = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, (tid, rec) in enumerate(pool.map(measure, sheet["target"])):
            records[tid] = rec
            if (i + 1) % 10 == 0 or i + 1 == len(sheet):
                print(f"  land cover / water / relief: {i + 1}/{len(sheet)} targets  [{time.time() - t0:.0f}s]", flush=True)
    for k, name in (("wc", "WorldCover"), ("gsw", "Surface Water"), ("dem", "Copernicus DEM")):
        errs = [r[f"{k}_error"] for r in records.values() if f"{k}_error" in r]
        if use[k] and errs:
            print(f"  {name}: {len(errs)} targets failed, e.g. {errs[0]}")
            if len(errs) == len(records):
                sources[name] = False

    # OpenStreetMap, one query per prospect area, two at a time (the free servers allow ~2 per user)
    if sources["OpenStreetMap"]:
        tasks = []
        for p, g in sheet.groupby("prospect", sort=False):
            geoms = [geo.get(t) for t in g["target"] if geo.get(t) is not None]
            if geoms:
                bb = gpd.GeoSeries(geoms, crs="EPSG:32643").total_bounds
                tasks.append((p, g, ll(box(bb[0] - OSM_MARGIN_M, bb[1] - OSM_MARGIN_M,
                                           bb[2] + OSM_MARGIN_M, bb[3] + OSM_MARGIN_M)).bounds))
        state = {"ok": 0, "fail": 0}
        lock = threading.Lock()

        def fetch(task):
            p, _, (w, s_, e, n) = task
            key = cache.key("osm", round(s_, 4), round(w, 4), round(n, 4), round(e, 4))
            js = cache.get("osm", key)
            if js is not None:
                return js
            with lock:
                if state["ok"] == 0 and state["fail"] >= 2:   # nothing has worked: stop trying
                    return None
            try:
                js = overpass_query(s_, w, n, e)
                cache.put("osm", key, js)
                time.sleep(1.0)   # be polite to the free Overpass servers
                with lock:
                    state["ok"] += 1
                return js
            except Exception as ex:
                print(f"  OSM {p}: {ex}", flush=True)
                with lock:
                    state["fail"] += 1
                return None

        osm_ok = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            for (p, g, _), js in zip(tasks, pool.map(fetch, tasks)):
                if js is None:
                    continue
                osm_ok = True
                f = osm_features(js)
                for _, r in g.iterrows():
                    geom = geo.get(r["target"])
                    if geom is not None:
                        peak = Point(TO_UTM.transform(r["peak_lon"], r["peak_lat"]))
                        records[r["target"]].update(osm_metrics(f, geom, peak))
        sources["OpenStreetMap"] = osm_ok
        print(f"  OpenStreetMap {'done' if osm_ok else 'unavailable'}  [{time.time() - t0:.0f}s]", flush=True)

    # Sentinel-2 in parallel
    if sources["Sentinel-2"]:
        todo = [{"target": t, "geometry": geo.get(t)} for t in sheet["target"] if geo.get(t) is not None]
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for t, res in zip(todo, pool.map(lambda tt: s2_for_target(tt, cache), todo)):
                records[t["target"]].update(res)
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"  Sentinel-2: {done}/{len(todo)} targets  [{time.time() - t0:.0f}s]", flush=True)

    # decide
    rec_df = pd.DataFrame.from_dict(records, orient="index")
    rec_df.index.name = "target"
    sheet = sheet.drop(columns=[c for c in rec_df.columns if c in sheet.columns]).merge(rec_df, on="target", how="left")
    decisions = [decide(rec_df.loc[t].to_dict() if t in rec_df.index else {}, f)
                 for t, f in zip(sheet["target"], sheet["flags"])]
    sheet["check_status"] = [d[0] for d in decisions]
    sheet["auto_confidence"] = [d[1] for d in decisions]
    sheet["auto_reasons"] = ["; ".join(d[2]) for d in decisions]
    sheet["needs_eye"] = [d[3] for d in decisions]
    wcols = [c for c in ("wc_water_all", "gsw_seasonal", "s2_wet_water", "gm_water_frac") if c in sheet.columns]
    sheet["water_max"] = sheet[wcols].max(axis=1) if wcols else np.nan

    front = ["order", "prospect", "best_tier", "target", "tier", "check_status", "auto_confidence", "needs_eye",
             "auto_reasons", "notes", "status", "flags", "look_for", "mean_pct", "stability", "dist_known_gold_km",
             "road_dist_m", "peak_lat", "peak_lon", "satellite_link"]
    front = [c for c in front if c in sheet.columns]
    rest = [c for c in sheet.columns if c not in front and not c.endswith("_chip")]
    chips = [c for c in sheet.columns if c.endswith("_chip")]
    sheet = sheet[front + rest + chips]
    sheet.to_csv(out / "desk_check_auto.csv", index=False, encoding="utf-8-sig")

    after = retier(sheet, prospects)
    after.to_csv(out / "prospects_after_check.csv", index=False)
    write_html(sheet, out / "desk_check_report.html", sources, after)
    print_summary(sheet, after, out, t0)


def print_summary(sheet: pd.DataFrame, after: pd.DataFrame, out: Path, t0: float) -> None:
    print(f"\nDesk check of {len(sheet)} targets  [{time.time() - t0:.0f}s]")
    print("  codes: " + ", ".join(f"{k} {v}" for k, v in sheet["check_status"].value_counts().items()))
    if "needs_eye" in sheet:
        print(f"  needs a human look: {int(sheet['needs_eye'].astype(bool).sum())} (shown first in the HTML report)")
    if len(after):
        ch = after[after["tier_before"] != after["tier_after"]]
        print(f"  prospects after check: {after['tier_after'].value_counts().to_dict()}; changed: {len(ch)}")
        with pd.option_context("display.width", 200, "display.max_colwidth", 40):
            cols = [c for c in ("prospect", "tier_before", "tier_after", "statuses", "dist_known_gold_km", "road_dist_m")
                    if c in after.columns]
            print(after[cols].to_string(index=False))
    print(f"\nOutputs in {out}: desk_check_auto.csv, desk_check_report.html, prospects_after_check.csv")


if __name__ == "__main__":
    main()
