# app_streamlit.py —— 懒加载版（未来班次过滤 + 30s 刷新 + Bus 折叠 + 稳定线路几何 + 官方配色）

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import urllib.request as urlreq

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ====== 实时工具（你的 Streamlit 版 utils）======
from utils_streamlit import (
    get_bus_schedule,
    get_subway_schedule,
    get_LIRR_schedule,
    get_MNR_schedule,
    color_interpolation,
)

# ---- 页面配置尽早设置 ----
st.set_page_config(page_title="Real Time Transportation Dashboard", layout="wide")

# ====== 可选：非阻塞自动刷新 ======
try:
    from streamlit_autorefresh import st_autorefresh

    _HAS_ST_AUTOR = True
except Exception:
    _HAS_ST_AUTOR = False


def safe_autorefresh(enabled: bool, interval_ms: int, key: str = "auto_refresh_key"):
    if enabled and _HAS_ST_AUTOR:
        st_autorefresh(interval=interval_ms, key=key)
    elif enabled and not _HAS_ST_AUTOR:
        st.caption(
            "Auto-refresh disabled (install `streamlit-autorefresh` to enable non-blocking refresh)."
        )


# ====== 路径 & 常量 ======
ROOT = Path(__file__).resolve().parent
GTFS_DIR = ROOT / "GTFS"

SUBFILES = [
    "bus_bronx",
    "bus_brooklyn",
    "bus_manhattan",
    "bus_queens",
    "bus_staten_island",
    "subway",
    "LIRR",
    "MNR",
    "bus_new_jersy",
]
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten_Island", "New_Jersy"]

BOROUGHS_COORDINATE_MAPPING = {
    "Bronx": [40.837048, -73.865433],
    "Brooklyn": [40.650002, -73.949997],
    "Manhattan": [40.776676, -73.971321],
    "Queens": [40.742054, -73.769417],
    "Staten_Island": [40.579021, -74.151535],
    "New_Jersy": [40.717, -74.1],
}

CITIBIKE_REGIONS = ["NYC District", "JC District", "Hoboken District"]
CITIBIKE_REGIONS_COLORING_DARK = {
    "NYC District": (0, 0, 139),
    "JC District": (0, 100, 0),
    "Hoboken District": (139, 0, 0),
}
CITIBIKE_REGIONS_COLORING_LIGHT = {
    "NYC District": (173, 216, 230),
    "JC District": (144, 238, 144),
    "Hoboken District": (255, 99, 71),
}

# MTA 官方地铁颜色（route_id -> hex 不带 #）
SUBWAY_OFFICIAL_COLORS = {
    "1": "EE352E",
    "2": "EE352E",
    "3": "EE352E",
    "4": "00933C",
    "5": "00933C",
    "6": "00933C",
    "7": "B933AD",
    "A": "2850AD",
    "C": "2850AD",
    "E": "2850AD",
    "B": "FF6319",
    "D": "FF6319",
    "F": "FF6319",
    "M": "FF6319",
    "G": "6CBE45",
    "J": "996633",
    "Z": "996633",
    "L": "A7A9AC",
    "N": "FCCC0A",
    "Q": "FCCC0A",
    "R": "FCCC0A",
    "W": "FCCC0A",
    "S": "808183",  # shuttles
}

# ======================
#   数据加载（静态）
# ======================
@st.cache_data(show_spinner=False)
def load_gtfs_tables(subdir: str):
    folder = GTFS_DIR / subdir
    need = ["routes.txt", "stop_times.txt", "stops.txt", "trips.txt"]
    if not (folder.exists() and all((folder / f).exists() for f in need)):
        return None
    # 关键：强制字符串，避免 stop_id/route_id 被转数字
    read = lambda f: pd.read_csv(folder / f, dtype=str)
    routes = read("routes.txt")
    stop_times = read("stop_times.txt")
    stops = read("stops.txt")
    trips = read("trips.txt")
    return routes, stop_times, stops, trips


# ---- 懒加载一个数据集（subway / LIRR / bus_xxx）并做 merge ----
@st.cache_resource(show_spinner=False)
def get_dataset(subdir: str) -> pd.DataFrame:
    tables = load_gtfs_tables(subdir)
    if tables is None:
        return pd.DataFrame(
            columns=[
                "route_id",
                "service_id",
                "trip_id",
                "arrival_time",
                "departure_time",
                "stop_sequence",
                "stop_id",
                "stop_name",
                "stop_lat",
                "stop_lon",
                "route_long_name",
                "color",
            ]
        )

    routes, stop_times, stops, trips = tables

    df = trips[["route_id", "service_id", "trip_id"]].merge(
        stop_times[
            ["trip_id", "arrival_time", "departure_time", "stop_sequence", "stop_id"]
        ],
        on="trip_id",
        how="left",
    ).merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left",
    ).merge(
        routes[["route_id", "route_long_name", "route_color"]],
        on="route_id",
        how="left",
    )

    # 颜色映射：地铁用官方配色，其它用 GTFS 自带
    route_color_mapping = (
        df.set_index("route_id")["route_color"]
        .fillna("000000")
        .astype(str)
        .apply(lambda x: "#" + x)
        .to_dict()
    )
    df["color"] = df["route_id"].map(route_color_mapping)

    if subdir == "subway":
        # 覆盖为官方颜色（找不到的仍用 GTFS 自带）
        def _subway_color(rid: str) -> str:
            rid = str(rid)
            if rid in SUBWAY_OFFICIAL_COLORS:
                return "#" + SUBWAY_OFFICIAL_COLORS[rid]
            return route_color_mapping.get(rid, "#FFFFFF")

        df["color"] = df["route_id"].astype(str).map(_subway_color)

    if subdir == "bus_new_jersy":
        df["color"] = "#00FF00"

    return df


# ---- route 列表懒加载 ----
@st.cache_data(show_spinner=False)
def get_subway_route_ids() -> list[str]:
    df = get_dataset("subway")
    if df.empty:
        return []
    return sorted(df["route_id"].astype(str).dropna().unique().tolist(), key=str)


@st.cache_data(show_spinner=False)
def get_lirr_route_ids() -> list[str]:
    df = get_dataset("LIRR")
    if df.empty:
        return []
    return sorted(df["route_id"].astype(str).dropna().unique().tolist(), key=str)


@st.cache_data(show_spinner=False)
def get_bus_route_ids(borough: str) -> list[str]:
    key = f"bus_{borough.lower()}"
    df = get_dataset(key)
    if df.empty:
        return []
    return sorted(df["route_id"].astype(str).dropna().unique().tolist(), key=str)


# =========================
#   实时 feed（缓存 30s & 仅保留未来班次）
# =========================
def filter_feed_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅保留“此刻之后”的最近一班：
      - route/stop_id 统一为 str
      - 解析 arrival/departure
      - 合成 when，并过滤 when >= now
      - 对每个 (route, stop) 选最早的未来时刻
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["route", "stop_id", "arrival_time", "departure_time"]
        )

    df = df.copy()
    df["route"] = df["route"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    now = pd.Timestamp.now()
    df["when"] = df["arrival_time"].fillna(df["departure_time"])
    df = df.dropna(subset=["when"])
    df = df[df["when"] >= now]

    if df.empty:
        return pd.DataFrame(
            columns=["route", "stop_id", "arrival_time", "departure_time"]
        )

    df = (
        df.sort_values("when")
        .groupby(["route", "stop_id"], as_index=False)
        .first()[["route", "stop_id", "arrival_time", "departure_time"]]
    )
    return df


@st.cache_data(ttl=30, show_spinner=False)
def fetch_subway_feed():
    return filter_feed_df(pd.DataFrame(get_subway_schedule()))


@st.cache_data(ttl=30, show_spinner=False)
def fetch_bus_feed():
    return filter_feed_df(pd.DataFrame(get_bus_schedule()))


@st.cache_data(ttl=30, show_spinner=False)
def fetch_lirr_feed():
    return filter_feed_df(pd.DataFrame(get_LIRR_schedule()))


@st.cache_data(ttl=30, show_spinner=False)
def fetch_mnr_feed():
    return filter_feed_df(pd.DataFrame(get_MNR_schedule()))


# =========================
#   预计算静态“线路几何”（基于 diff 的切段 + 清洗小段）
# =========================
def precompute_route_lines_df(df: pd.DataFrame) -> dict[str, list[pd.DataFrame]]:
    """
    对给定 df（某个 subdir）构造：
      route_id -> [segment_df, ...]
    """
    if df is None or df.empty:
        return {}

    base = df[
        [
            "route_id",
            "trip_id",
            "stop_sequence",
            "stop_id",
            "stop_lat",
            "stop_lon",
            "route_long_name",
            "color",
            "stop_name",
        ]
    ].dropna(subset=["route_id", "trip_id", "stop_id", "stop_sequence"]).copy()

    for col in ["route_id", "trip_id", "stop_id"]:
        base[col] = base[col].astype(str)

    # 选每个 route 停站数最多的 trip 作为代表
    counts = (
        base.groupby(["route_id", "trip_id"])["stop_id"]
        .nunique()
        .reset_index(name="n_stops")
    )
    idx = (
        counts.sort_values(["route_id", "n_stops"], ascending=[True, False])
        .groupby("route_id")
        .head(1)
    )
    rep = base.merge(idx[["route_id", "trip_id"]], on=["route_id", "trip_id"], how="inner")

    res: dict[str, list[pd.DataFrame]] = {}
    for (rid, _tid), g in rep.groupby(["route_id", "trip_id"], sort=False):
        g = g.copy()
        g["stop_sequence"] = pd.to_numeric(g["stop_sequence"], errors="coerce")
        g = (
            g.dropna(subset=["stop_sequence"])
            .sort_values("stop_sequence")
            .reset_index(drop=True)
        )

        g["stop_lat"] = pd.to_numeric(g["stop_lat"], errors="coerce")
        g["stop_lon"] = pd.to_numeric(g["stop_lon"], errors="coerce")
        g = g.dropna(subset=["stop_lat", "stop_lon"])

        # 基于 stop_sequence diff 切段
        cut = g["stop_sequence"].diff().fillna(1) != 1
        seg_id = cut.cumsum()

        subs: list[pd.DataFrame] = []
        for _, seg in g.groupby(seg_id, sort=False):
            # 去掉连续重复坐标
            seg = seg.loc[
                ~(
                    seg["stop_lat"].diff().fillna(0).eq(0)
                    & seg["stop_lon"].diff().fillna(0).eq(0)
                )
            ]
            if len(seg) <= 2:  # 太短的段不要
                continue
            subs.append(seg)

        res[str(rid)] = subs

    return res


# 懒加载线路几何
@st.cache_resource(show_spinner=False)
def get_subway_lines() -> dict[str, list[pd.DataFrame]]:
    df = get_dataset("subway")
    return precompute_route_lines_df(df)


@st.cache_resource(show_spinner=False)
def get_lirr_lines() -> dict[str, list[pd.DataFrame]]:
    df = get_dataset("LIRR")
    return precompute_route_lines_df(df)


@st.cache_resource(show_spinner=False)
def get_bus_lines(borough: str) -> dict[str, list[pd.DataFrame]]:
    key = f"bus_{borough.lower()}"
    df = get_dataset(key)
    return precompute_route_lines_df(df)


# =========================
#   画图工具（带图例组开关）
# =========================
def _default_hover(sub_df: pd.DataFrame) -> list[str]:
    return [f"Stop: {name}" for name in sub_df["stop_name"].astype(str).tolist()]


def _with_arrival_hover(
    sub_df: pd.DataFrame, schedule_map: dict[tuple[str, str], str], route_id: str
) -> list[str]:
    texts = []
    stops = sub_df[["stop_id", "stop_name"]].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    for _, row in stops.iterrows():
        key = (str(route_id), str(row["stop_id"]))
        arr = schedule_map.get(key, "N/A")
        texts.append(f"Stop: {row['stop_name']}<br>Next arrival: {arr}")
    return texts


def _pick_color_from_subs(subs: list[pd.DataFrame]) -> str:
    for s in subs:
        try:
            c = str(s["color"].iloc[0])
            if c and c != "#000000":
                return c
        except Exception:
            continue
    return "blue"


def _base_fig(center=(40.8, -74), zoom=10) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        mapbox={
            "center": {"lat": center[0], "lon": center[1]},
            "style": "carto-darkmatter",
            "zoom": zoom,
        },
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
        legend=dict(
            title="Routes",
            groupclick="togglegroup",
            bgcolor="rgba(0, 0, 0, 0)",
        ),
    )
    return fig


def _add_lines_to_fig(
    fig: go.Figure,
    subs: list[pd.DataFrame],
    color: str,
    show_markers: bool,
    hover_text_builder,
    route_id: str,
    route_label: str | None = None,
):
    if not subs:
        return
    line_color = color if (isinstance(color, str) and color and color != "#000000") else "blue"
    route_label = route_label or f"route {route_id}"

    first = True
    for s in subs:
        fig.add_trace(
            go.Scattermapbox(
                lon=s["stop_lon"],
                lat=s["stop_lat"],
                mode="lines",
                line=dict(width=3, color=line_color),
                hoverinfo="text",
                text=hover_text_builder(s),
                legendgroup=f"route-{route_id}",
                showlegend=first,
                name=route_label,
            )
        )
        first = False
        if show_markers:
            fig.add_trace(
                go.Scattermapbox(
                    lon=s["stop_lon"],
                    lat=s["stop_lat"],
                    mode="markers",
                    marker=dict(symbol="circle", size=4, color="white"),
                    hoverinfo="skip",
                    legendgroup=f"route-{route_id}",
                    showlegend=False,
                    name=route_label,
                )
            )


# =========================
#   各图层构图
# =========================
def build_subway_figure(
    selected_routes: list[str], show_arrival: bool, show_stops: bool
) -> go.Figure:
    fig = _base_fig(center=(40.78, -73.97), zoom=10)
    lines = get_subway_lines()
    routes = selected_routes or list(lines.keys())

    schedule_map: dict[tuple[str, str], str] = {}
    if show_arrival:
        sched = fetch_subway_feed()
        if not sched.empty:
            sched = sched[sched["route"].astype(str).isin([str(r) for r in routes])]
            if not sched.empty:
                sched["stop_id"] = sched["stop_id"].astype(str)
                schedule_map = {
                    (str(r), str(s)): str(a)
                    for r, s, a in zip(
                        sched["route"], sched["stop_id"], sched["arrival_time"]
                    )
                }

    for rid in routes:
        subs = lines.get(str(rid), [])
        color = _pick_color_from_subs(subs)
        hover_builder = (
            (lambda s, _rid=str(rid): _with_arrival_hover(s, schedule_map, _rid))
            if show_arrival
            else _default_hover
        )
        _add_lines_to_fig(
            fig,
            subs,
            color,
            show_stops,
            hover_builder,
            route_id=str(rid),
            route_label=f"Subway {rid}",
        )
    return fig


def build_bus_borough_figure(
    borough: str, selected_routes: list[str], show_arrival: bool, show_stops: bool
) -> go.Figure:
    center = BOROUGHS_COORDINATE_MAPPING[borough]
    fig = _base_fig(center=(center[0], center[1]), zoom=10)
    lines_dict = get_bus_lines(borough)
    routes = selected_routes or list(lines_dict.keys())

    schedule_map: dict[tuple[str, str], str] = {}
    if show_arrival:
        sched = fetch_bus_feed()
        if not sched.empty:
            sched = sched[sched["route"].astype(str).isin([str(r) for r in routes])]
            if not sched.empty:
                sched["stop_id"] = sched["stop_id"].astype(str)
                schedule_map = {
                    (str(r), str(s)): str(a)
                    for r, s, a in zip(
                        sched["route"], sched["stop_id"], sched["arrival_time"]
                    )
                }

    for rid in routes:
        subs = lines_dict.get(str(rid), [])
        color = _pick_color_from_subs(subs)
        hover_builder = (
            (lambda s, _rid=str(rid): _with_arrival_hover(s, schedule_map, _rid))
            if show_arrival
            else _default_hover
        )
        _add_lines_to_fig(
            fig,
            subs,
            color,
            show_stops,
            hover_builder,
            route_id=str(rid),
            route_label=f"Bus {rid}",
        )
    return fig


def build_lirr_figure(
    selected_routes: list[str], show_arrival: bool, show_stops: bool
) -> go.Figure:
    fig = _base_fig(center=(40.8, -74), zoom=10)
    lines = get_lirr_lines()
    routes = selected_routes or list(lines.keys())

    schedule_map: dict[tuple[str, str], str] = {}
    if show_arrival:
        sched = fetch_lirr_feed()
        if not sched.empty:
            sched = sched[sched["route"].astype(str).isin([str(r) for r in routes])]
            if not sched.empty:
                sched["stop_id"] = sched["stop_id"].astype(str)
                schedule_map = {
                    (str(r), str(s)): str(a)
                    for r, s, a in zip(
                        sched["route"], sched["stop_id"], sched["arrival_time"]
                    )
                }

    for rid in routes:
        subs = lines.get(str(rid), [])
        color = _pick_color_from_subs(subs)
        hover_builder = (
            (lambda s, _rid=str(rid): _with_arrival_hover(s, schedule_map, _rid))
            if show_arrival
            else _default_hover
        )
        _add_lines_to_fig(
            fig,
            subs,
            color,
            show_stops,
            hover_builder,
            route_id=str(rid),
            route_label=f"LIRR {rid}",
        )
    return fig


# =========== Citibike ===========
@st.cache_data(ttl=120, show_spinner=False)
def citibike_station_data() -> pd.DataFrame:
    try:
        info = json.load(
            urlreq.urlopen(
                "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
            )
        )
        status = json.load(
            urlreq.urlopen("https://gbfs.citibikenyc.com/gbfs/en/station_status.json")
        )
        regions = json.load(
            urlreq.urlopen("https://gbfs.citibikenyc.com/gbfs/en/system_regions.json")
        )
    except Exception:
        return pd.DataFrame(
            columns=[
                "name",
                "lat",
                "lon",
                "capacity",
                "region_id",
                "region_name",
                "num_docks_available",
                "num_ebikes_available",
                "num_bikes_available",
                "last_reported",
            ]
        )

    info_df = pd.DataFrame(info["data"]["stations"]).set_index("station_id")[
        ["name", "lat", "lon", "capacity", "region_id"]
    ]
    status_df = pd.DataFrame(status["data"]["stations"]).set_index("station_id")[
        [
            "num_docks_available",
            "num_bikes_disabled",
            "num_ebikes_available",
            "num_bikes_available",
            "num_docks_disabled",
            "is_renting",
            "is_returning",
            "last_reported",
            "is_installed",
        ]
    ]
    regions_df = pd.DataFrame(regions["data"]["regions"]).rename(
        columns={"name": "region_name"}
    )
    return (
        info_df.merge(status_df, left_index=True, right_index=True)
        .merge(
            regions_df[["region_id", "region_name"]],
            left_on="region_id",
            right_on="region_id",
        )
    )


def _citibike_row_to_color(row: pd.Series) -> str:
    dark = CITIBIKE_REGIONS_COLORING_DARK[row["region_name"]]
    light = CITIBIKE_REGIONS_COLORING_LIGHT[row["region_name"]]
    ratio = min(int(row["num_bikes_available"]), 80) / 80
    return f"rgba{color_interpolation(dark, light, ratio)}"


def build_citibike_figure(selected_regions: list[str]) -> go.Figure:
    fig = _base_fig(center=(40.776676, -73.971321), zoom=11)
    cb = citibike_station_data()
    if cb.empty:
        st.warning("Citibike API unavailable, please retry later.")
        return fig

    cb = cb.sort_values(by=["lat", "lon", "last_reported"], ascending=False).drop_duplicates(
        ["lat", "lon"]
    )
    cb["color"] = cb.apply(_citibike_row_to_color, axis=1)
    cb["last_reported"] = cb["last_reported"].apply(
        lambda x: datetime.fromtimestamp(int(x))
    )
    for rg in selected_regions:
        sub = cb[cb["region_name"] == rg]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattermapbox(
                lon=sub["lon"],
                lat=sub["lat"],
                mode="markers",
                marker=dict(size=10, color=sub["color"]),
                text=sub.apply(
                    lambda x: (
                        f"Name: {x['name']}<br>"
                        f"Docks: {x['num_docks_available']}<br>"
                        f"eBikes: {x['num_ebikes_available']}<br>"
                        f"Bikes: {x['num_bikes_available']}<br>"
                        f"Last: {x['last_reported']}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                legendgroup=f"citibike-{rg}",
                showlegend=True,
                name=f"Citibike {rg}",
            )
        )
    return fig


# =========================
#         UI
# =========================
st.title("Real Time Transportation Dashboard")

with st.sidebar:
    st.subheader("Choose a map to display")
    map_choice = st.radio("Layer", options=["subway", "LIRR", "bus", "citibike"], index=0)

    bus_borough = None
    if map_choice == "bus":
        bus_borough = st.selectbox("Bus borough", BOROUGHS, index=2)

    st.divider()
    st.subheader("Rendering options")
    show_arrival = st.checkbox("Show next-arrival time (slower)", value=False)
    show_stops = st.checkbox("Show stop markers (slowest)", value=False)

    st.divider()
    auto_refresh = st.toggle("Auto refresh maps (30s)", value=True)
    if _HAS_ST_AUTOR:
        st.caption("Auto-refresh by `streamlit-autorefresh` (non-blocking).")

# 图层专属筛选
with st.sidebar:
    selected_subway: list[str] = []
    selected_bus: list[str] = []
    selected_lirr: list[str] = []
    selected_regions: list[str] = []

    if map_choice == "subway":
        subway_routes = get_subway_route_ids()
        selected_subway = st.multiselect("Subway routes", subway_routes, default=[])

    elif map_choice == "bus":
        _borough = bus_borough or "Manhattan"
        bus_routes = get_bus_route_ids(_borough)
        selected_bus = st.multiselect(f"{_borough} bus routes", bus_routes, default=[])

    elif map_choice == "LIRR":
        lirr_routes = get_lirr_route_ids()
        selected_lirr = st.multiselect("LIRR routes", lirr_routes, default=[])

    elif map_choice == "citibike":
        selected_regions = st.multiselect(
            "Citibike regions", CITIBIKE_REGIONS, default=CITIBIKE_REGIONS
        )

    # 立即刷新按钮 + 上次更新时间
    st.divider()
    cols = st.columns([1, 1.4])
    with cols[0]:
        if st.button("🔄 Refresh now"):
            # 只清实时 & citibike cache
            fetch_subway_feed.clear()
            fetch_bus_feed.clear()
            fetch_lirr_feed.clear()
            fetch_mnr_feed.clear()
            citibike_station_data.clear()
            st.experimental_rerun()
    with cols[1]:
        st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")

# 非阻塞自动刷新（30s）
safe_autorefresh(enabled=auto_refresh, interval_ms=30 * 1000)

# ---------- 绘制 ----------
try:
    if map_choice == "subway":
        fig = build_subway_figure(selected_subway, show_arrival, show_stops)
    elif map_choice == "LIRR":
        fig = build_lirr_figure(selected_lirr, show_arrival, show_stops)
    elif map_choice == "bus":
        _borough = bus_borough or "Manhattan"
        fig = build_bus_borough_figure(_borough, selected_bus, show_arrival, show_stops)
    else:  # citibike
        fig = build_citibike_figure(selected_regions or CITIBIKE_REGIONS)

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
except Exception as e:
    st.exception(e)

# 底部统计
with st.sidebar:
    st.divider()
    _bn = bus_borough or "Manhattan"
    subway_lines = get_subway_lines()
    lirr_lines = get_lirr_lines()
    bus_routes_for_bn = get_bus_route_ids(_bn)
    st.caption(
        f"subway routes: {len(subway_lines)} | "
        f"LIRR routes: {len(lirr_lines)} | "
        f"bus({_bn}): {len(bus_routes_for_bn)}"
    )
