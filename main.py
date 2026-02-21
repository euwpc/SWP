import os, time, datetime, threading, warnings, concurrent.futures
import requests, xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

UPDATE_INTERVAL = 300        # seconds (5 min); change to 600 for 10 min. Changing it to less than 5 minutes may result in newer data not loading, therefore not recommended.
DPI             = 200
GRID_STEP       = 0.012      # ~1 km interpolation grid
SIGMA           = 2.5        # Gaussian smoothing (grid cells)
SMHI_BASE       = "https://opendata-download-metobs.smhi.se/api/version/1.0"
_HDR = {"User-Agent": "SwedenWeatherMap/4.0 (python/requests)"}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_EXTENT_S   = [10.4, 19.6, 55.2, 59.7]   # Södra Sverige
_EXTENT_C   = [10.8, 21.0, 58.8, 63.6]   # Mellersta Sverige
_EXTENT_N   = [11.2, 24.6, 62.3, 69.2]   # Norra Sverige
_EXTENT_ALL = [10.4, 24.6, 55.2, 69.2]   # Hela landet

EXTENT       = _EXTENT_ALL
REGION_LABEL = "Hela landet"

_SHPCACHE = {}

def _get_sweden_geom():
    if "SE" in _SHPCACHE:
        return _SHPCACHE["SE"]
    shp = shpreader.natural_earth(resolution="10m", category="cultural",
                                  name="admin_0_countries")
    reader = shpreader.Reader(shp)
    se_geoms, ot_geoms = [], []
    for rec in reader.records():
        iso = rec.attributes.get("ISO_A2", "")
        a3  = rec.attributes.get("ADM0_A3", "")
        g   = rec.geometry
        if iso == "SE" or a3 == "SWE":
            se_geoms.append(g)
        else:
            ot_geoms.append(g)
    se = unary_union(se_geoms) if se_geoms else None
    ot = unary_union(ot_geoms) if ot_geoms else None
    _SHPCACHE["SE"] = (se, ot)
    return se, ot


def parse_qml(filename, vmin, vmax):
    path = os.path.join(SCRIPT_DIR, filename)
    root = ET.parse(path).getroot()
    vals, cols = [], []
    for item in root.findall(".//colorrampshader/item"):
        v = float(item.get("value"))
        h = item.get("color").lstrip("#")
        r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
        vals.append(v); cols.append((r, g, b))
    vals = np.array(vals); cols = np.array(cols)
    idx  = np.argsort(vals); vals, cols = vals[idx], cols[idx]
    pos  = np.clip((vals - vmin) / (vmax - vmin), 0., 1.)
    cmap = LinearSegmentedColormap.from_list("qml", list(zip(pos, cols)), N=2048)
    return cmap, Normalize(vmin=vmin, vmax=vmax)

def _bc(stops, v0, v1):
    pos = [max(0., min(1., (v - v0) / (v1 - v0))) for v, _ in stops]
    cm  = LinearSegmentedColormap.from_list(
        "fb", list(zip(pos, [c for _, c in stops])), N=2048)
    return cm, Normalize(vmin=v0, vmax=v1)

def load_cmaps():
    c = {}
    try:
        c["temp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except Exception as e:
        print(f"  [warn] temp QML: {e}")
        c["temp"] = _bc([(-40,"#ff6eff"),(-20,"#32007f"),(-10,"#259aff"),
                          (0,"#d9ecff"),(10,"#52ca0b"),(20,"#f4bd0b"),
                          (30,"#af0f14"),(45,"#c5c5c5")], -40, 45)
    try:
        c["wind"] = parse_qml("wind_gust_color_table.qml", 0, 50)
    except Exception as e:
        print(f"  [warn] wind QML: {e}")
        c["wind"] = _bc([(0,"#ffffff"),(5,"#aad4ff"),(10,"#3c96f5"),
                          (15,"#ffdd00"),(20,"#ffa000"),
                          (30,"#e11400"),(50,"#8c645a")], 0, 50)
    c["gust"] = c["wind"]
    try:
        c["pres"] = parse_qml("pressure_color_table.qml", 890, 1064)
    except Exception as e:
        print(f"  [warn] pres QML: {e}")
        c["pres"] = _bc([(965,"#32007f"),(990,"#91ccff"),(1000,"#07a127"),
                          (1013,"#f3fb01"),(1030,"#f4520b"),(1050,"#f0a0a0")],
                         960, 1055)
    try:
        c["prec"] = parse_qml("precipitation_color_table.qml", 0, 125)
    except Exception as e:
        print(f"  [warn] prec QML: {e}")
        c["prec"] = _bc([(0,"#f0f0f0"),(0.1,"#aaddff"),(1,"#0482ff"),
                          (5,"#1acf05"),(10,"#ff7f27"),
                          (20,"#bf0000"),(50,"#64007f")], 0, 50)
    c["hum"]  = _bc([(0,"#ffffff"),(20,"#d0eaff"),(50,"#55a3e0"),
                      (80,"#084a90"),(100,"#021860")], 0, 100)
    try:
        c["dewp"] = parse_qml("temperature_color_table_high.qml", -40, 50)
    except Exception:
        c["dewp"] = c["temp"]
    return c


_PARAMS = [
    ("temp", 1,  "latest-hour"),
    ("wdir", 3,  "latest-hour"),
    ("wind", 4,  "latest-hour"),
    ("hum",  6,  "latest-hour"),
    ("pres", 9,  "latest-hour"),
    ("gust", 21, "latest-hour"),
    ("dewp", 39, "latest-hour"),
    ("prec", 14, "latest-hour"),
]


def _fetch_one_json(key, param_id, period):
    """Fetch data.json for one parameter, return list of station dicts."""
    url = (f"{SMHI_BASE}/parameter/{param_id}"
           f"/station-set/all/period/{period}/data.json")
    try:
        r = requests.get(url, headers=_HDR, timeout=35)
        if r.status_code == 404 and period == "latest-hour":
            url2 = url.replace("latest-hour", "latest-day")
            r = requests.get(url2, headers=_HDR, timeout=35)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [SMHI] {key:4s} param {param_id:2d} failed: {e}")
        return []

    stations = data.get("station", [])
    result = []
    count = 0
    for s in stations:
        lat = s.get("latitude")
        lon = s.get("longitude")
        if lat is None or lon is None:
            continue
        val_list = s.get("value", [])
        if not val_list:
            continue
        raw = val_list[-1].get("value")
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if key == "prec" and v < 0:
            continue
        result.append({
            "key":  int(s.get("key", 0)),
            "name": s.get("name", "?"),
            "lat":  float(lat),
            "lon":  float(lon),
            "val":  v,
        })
        count += 1

    print(f"  [SMHI] {key:4s} param {param_id:2d}: {count} stations with data")
    return result


def fetch(extent):
    # Fetch all stations from SMHI regardless of region.
    # A generous margin is kept so that interpolation is smooth right to the
    # edges of any sub-region; render_one handles what actually gets drawn.
    FETCH_MARGIN = 1.5
    lon0, lon1, lat0, lat1 = extent

    raw = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(_fetch_one_json, k, pid, per): k
            for k, pid, per in _PARAMS
        }
        for fut in concurrent.futures.as_completed(futs):
            k = futs[fut]
            raw[k] = fut.result()

    station_pos = {}
    for k, slist in raw.items():
        for s in slist:
            pos = (round(s["lat"], 4), round(s["lon"], 4))
            if pos not in station_pos:
                station_pos[pos] = {"name": s["name"],
                                    "lat": s["lat"], "lon": s["lon"]}

    val_by_key = {}
    for k, slist in raw.items():
        lut = {}
        for s in slist:
            pos = (round(s["lat"], 4), round(s["lon"], 4))
            lut[pos] = s["val"]
        val_by_key[k] = lut

    stations = []
    for pos, info in station_pos.items():
        lat, lon = info["lat"], info["lon"]
        # Include stations within the region plus a margin for interpolation.
        # Stations outside the visible extent will still inform the grid but
        # won't have dots/labels drawn (that filter lives in render_one).
        if not (lon0 - FETCH_MARGIN <= lon <= lon1 + FETCH_MARGIN and
                lat0 - FETCH_MARGIN <= lat <= lat1 + FETCH_MARGIN):
            continue
        s = {"name": info["name"], "lat": lat, "lon": lon}
        for k in ("temp", "wind", "wdir", "gust", "hum", "prec", "pres", "dewp"):
            s[k] = val_by_key.get(k, {}).get(pos, np.nan)
        s["wind_dir"] = s.pop("wdir")
        stations.append(s)

    time_str = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")
    print(f"  Totalt / Total stations in region: {len(stations)}  |  {time_str}")
    return stations, time_str


def make_grid(extent):
    lons = np.arange(extent[0], extent[1] + GRID_STEP, GRID_STEP)
    lats = np.arange(extent[2], extent[3] + GRID_STEP, GRID_STEP)
    return np.meshgrid(lons, lats)

def interpolate(stations, key, gx, gy):
    ok = [s for s in stations if not np.isnan(s.get(key, np.nan))]
    if len(ok) < 4:
        return None
    pts = np.array([(s["lon"], s["lat"]) for s in ok])
    vs  = np.array([s[key]               for s in ok])
    zi  = griddata(pts, vs, (gx, gy), method="linear")
    znn = griddata(pts, vs, (gx, gy), method="nearest")
    zi  = np.where(np.isnan(zi), znn, zi)
    return gaussian_filter(zi, sigma=SIGMA)

_PE  = [mpe.withStroke(linewidth=3, foreground="white")]
_PE2 = [mpe.withStroke(linewidth=4, foreground="white")]


def render_one(stations, key, cmap, norm, title, unit, fmt,
               time_str, gx, gy, outfile, wind_arrows=False):
    lon0, lon1, lat0, lat1 = EXTENT
    grid = interpolate(stations, key, gx, gy)

    # Only show dots, labels and min/max for stations inside the visible extent.
    ok = [s for s in stations
          if not np.isnan(s.get(key, np.nan))
          and lon0 - 0.1 <= s["lon"] <= lon1 + 0.1
          and lat0 - 0.1 <= s["lat"] <= lat1 + 0.1]

    if ok:
        vmin_obs = min(s[key] for s in ok)
        vmax_obs = max(s[key] for s in ok)
        s_min    = min(ok, key=lambda s: s[key])
        s_max    = max(ok, key=lambda s: s[key])
    else:
        vmin_obs = vmax_obs = 0
        s_min = s_max = None

    se_geom, ot_geom = _get_sweden_geom()

    lon_span = lon1 - lon0
    lat_span = lat1 - lat0
    cos_mid  = np.cos(np.radians((lat0 + lat1) / 2))
    phys_ratio = lat_span / (lon_span * cos_mid)

    fig_w  = 10.0
    fig_h  = fig_w * phys_ratio * 0.88 + 0.9   
    fig_h  = max(8.0, min(fig_h, 22.0))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white", dpi=DPI)

    top_in  = 0.55   
    bot_in  = 0.10   
    left_in = 0.10
    cb_in   = 1.10   

    l = left_in / fig_w
    b = bot_in  / fig_h
    w = (fig_w - left_in - cb_in) / fig_w
    h = (fig_h - top_in  - bot_in) / fig_h

    ax = fig.add_axes([l, b, w, h], projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("#cce4f0")   

    

    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="white", edgecolor="none", zorder=2)

    if grid is not None and se_geom is not None:
        pcm = ax.pcolormesh(gx, gy, np.ma.masked_invalid(grid),
                            cmap=cmap, norm=norm, shading="auto",
                            transform=ccrs.PlateCarree(),
                            rasterized=True, zorder=3)
        
        from cartopy.mpl.patch import geos_to_path
        import matplotlib.patches as mpatches
        clip_path = geos_to_path(se_geom)
        from matplotlib.path import Path
        import itertools
        verts = list(itertools.chain.from_iterable(
            p.vertices for p in clip_path))
        codes = list(itertools.chain.from_iterable(
            p.codes  for p in clip_path))
        compound = Path(verts, codes)
        patch = mpatches.PathPatch(compound, transform=ccrs.PlateCarree()
                                   ._as_mpl_transform(ax))
        pcm.set_clip_path(patch)

    if ot_geom is not None:
        ax.add_geometries([ot_geom], ccrs.PlateCarree(),
                          facecolor="white", edgecolor="none", zorder=4)

    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="#cce4f0", edgecolor="#4488aa",
                   linewidth=0.4, zorder=5)

    ax.coastlines(resolution="10m", linewidth=1.3, color="#111111", zorder=6)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   linestyle="-", edgecolor="#333333", linewidth=1.0, zorder=6)

    admin1 = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m",
        facecolor="none", edgecolor="#777777", linewidth=0.4)
    ax.add_feature(admin1, zorder=6)

    if wind_arrows:
        for s in ok:
            slon, slat = s["lon"], s["lat"]
            wd = s.get("wind_dir", np.nan)
            if np.isnan(wd) or s[key] < 0.5:
                continue
            wr = np.radians(wd)
            u  = -s[key] * np.sin(wr) * 0.08
            v  = -s[key] * np.cos(wr) * 0.08 * cos_mid
            ax.annotate("",
                xy    =(slon + u, slat + v),
                xytext=(slon,     slat),
                arrowprops=dict(arrowstyle="-|>", color="#111111",
                                lw=0.8, mutation_scale=7),
                transform=ccrs.PlateCarree(), zorder=10)

    for s in ok:
        slon, slat = s["lon"], s["lat"]
        ax.plot(slon, slat, "o", color="#222222", ms=2.5, mec="none",
                transform=ccrs.PlateCarree(), zorder=11)
        ax.text(slon + 0.06, slat + 0.04, fmt.format(s[key]),
                fontsize=6.0, fontweight="bold", color="#111111",
                path_effects=_PE, transform=ccrs.PlateCarree(), zorder=12)

    if s_min and s_max:
        for yv, yn, val, name, col in [
            (0.17, 0.12, vmax_obs, s_max["name"], "#cc0000"),
            (0.07, 0.02, vmin_obs, s_min["name"], "#0055cc"),
        ]:
            ax.text(0.012, yv, f"{fmt.format(val)}{unit}",
                    transform=ax.transAxes, fontsize=12,
                    fontweight="bold", color=col,
                    path_effects=_PE2, zorder=20)
            ax.text(0.012, yn, name,
                    transform=ax.transAxes, fontsize=7.5,
                    fontweight="bold", color=col,
                    path_effects=_PE2, zorder=20)

    cb_bar_w  = 0.14   
    cb_bar_h  = h * fig_h * 0.52   
    cb_left_in = left_in + w * fig_w + 0.25
    cb_bot_in  = bot_in  + h * fig_h * 0.24

    cax = fig.add_axes([cb_left_in / fig_w,
                        cb_bot_in  / fig_h,
                        cb_bar_w   / fig_w,
                        cb_bar_h   / fig_h])
    cb  = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cax, orientation="vertical", extend="both")
    cb.set_label(unit, fontsize=7, rotation=270, labelpad=9)
    cb.ax.tick_params(labelsize=6, length=2, pad=2)
    cb.outline.set_linewidth(0.4)

    ax.set_title(
        f"Sverige – {REGION_LABEL}  •  {title}\n{time_str}",
        fontsize=10, fontweight="bold", pad=5,
        loc="center", color="#111111")

    
    ax.annotate(
        "Datakälla: SMHI Open Data  •  opendata-download-metobs.smhi.se",
        xy=(0.5, 0.005), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=7.0, color="#888888", zorder=25)

    fig.savefig(outfile, dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"  Saved  {os.path.basename(outfile)}")


PANELS = [
    ("temp", "Lufttemperatur 2 m",             "°C",  "{:.1f}", False, "map_1_temperature.png"),
    ("wind", "Vindhastighet 10 m",             "m/s", "{:.1f}", True,  "map_2_wind_speed.png"),
    ("gust", "Byvind – max",                   "m/s", "{:.1f}", True,  "map_3_wind_gust.png"),
    ("pres", "Lufttryck red. till havsnivå",   "hPa", "{:.1f}", False, "map_4_pressure.png"),
    ("hum",  "Relativ luftfuktighet",          "%",   "{:.0f}", False, "map_5_humidity.png"),
    ("prec", "Nederbördsmängd (mm/15 min)",      "mm",  "{:.1f}", False, "map_6_precipitation.png"),
    ("dewp", "Daggpunkt 2 m",                  "°C",  "{:.1f}", False, "map_7_dewpoint.png"),
]



def run_once(cmaps, gx, gy):
    try:
        stations, time_str = fetch(EXTENT)
    except Exception as e:
        print(f"  Fetch error: {e}")
        return
    for (key, title, unit, fmt, arrows, fname) in PANELS:
        outfile = os.path.join(SCRIPT_DIR, fname)
        try:
            render_one(stations, key, *cmaps[key],
                       title, unit, fmt, time_str, gx, gy, outfile,
                       wind_arrows=arrows)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  Error {fname}: {e}")

def loop(cmaps, gx, gy):
    while True:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{now}] Uppdaterar / Updating …")
        run_once(cmaps, gx, gy)
        print(f"  Nästa uppdatering om {UPDATE_INTERVAL // 60} min.")
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        Väderkartor för Sverige – Välj region         ║")
    print("║        Swedish Weather Maps  – Choose region         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  1)  Södra Sverige      (Southern Sweden)            ║")
    print("║  2)  Mellersta Sverige  (Central Sweden)             ║")
    print("║  3)  Norra Sverige      (Northern Sweden)            ║")
    print("║  4)  Hela landet        (Whole country)              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    REGION_MAP = {
        "1": (_EXTENT_S,   "Södra Sverige"),
        "2": (_EXTENT_C,   "Mellersta Sverige"),
        "3": (_EXTENT_N,   "Norra Sverige"),
        "4": (_EXTENT_ALL, "Hela landet"),
    }

    while True:
        try:
            choice = input("  Välj / Choose [1/2/3/4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped."); raise SystemExit

        if choice in REGION_MAP:
            EXTENT, REGION_LABEL = REGION_MAP[choice]
            break
        else:
            print("  Ange 1, 2, 3 eller 4.  /  Please enter 1, 2, 3 eller 4.")

    print()
    print(f"  Region: {REGION_LABEL}")
    print(f"  Auto-uppdatering var {UPDATE_INTERVAL // 60} min. Ctrl+C för att avsluta.")
    print()

    print("  Laddar kartdata / Loading map data …")
    _get_sweden_geom()

    cmaps  = load_cmaps()
    gx, gy = make_grid(EXTENT)
    run_once(cmaps, gx, gy)

    t = threading.Thread(target=loop, args=(cmaps, gx, gy), daemon=True)
    t.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped.")
