# utils_streamlit.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from google.transit import gtfs_realtime_pb2
from datetime import datetime


# ---------------------------
# 配置与工具
# ---------------------------
ROOT = Path(__file__).resolve().parent
GTFS_DIR = ROOT / "GTFS"

# 允许用环境变量覆盖本地 key（更安全）
ENV_SUBWAY_KEY = os.getenv("MTA_SUBWAY_API_KEY")
ENV_BUS_KEY = os.getenv("MTA_BUS_API_KEY")

SUBWAY_KEY_PATH = GTFS_DIR / "subway_API_Key.txt"
BUS_KEY_PATH = GTFS_DIR / "bus_API_Key.txt"

# 网络请求参数
TIMEOUT = 12  # 秒
SUBWAY_HEADERS = {"x-api-key": (ENV_SUBWAY_KEY or SUBWAY_KEY_PATH.read_text(encoding="utf-8").strip())}
BUS_HEADERS = {}  # OBANYC 的 bus GTFS-rt 用 querystring 带 key，不用 header

def _safe_read_key(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def _get_subway_key() -> str:
    return ENV_SUBWAY_KEY or _safe_read_key(SUBWAY_KEY_PATH)

def _get_bus_key() -> str:
    return ENV_BUS_KEY or _safe_read_key(BUS_KEY_PATH)

def _ts_to_str(ts: Optional[int]) -> Optional[str]:
    """时间戳 → 字符串；字段缺失返回 None。"""
    try:
        if ts is None or int(ts) <= 0:
            return None
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _append_stop_time(rows: List[Dict], route_id: str, stop_update) -> None:
    """
    安全抽取 arrival/departure，字段缺失时允许为 None，统一 push。
    """
    arr = stop_update.arrival.time if stop_update.HasField("arrival") else None
    dep = stop_update.departure.time if stop_update.HasField("departure") else None
    rows.append(
        {
            "route": route_id,
            "arrival_time": _ts_to_str(arr),
            "departure_time": _ts_to_str(dep),
            "stop_id": getattr(stop_update, "stop_id", None),
        }
    )

def color_interpolation(dark_color: Tuple[int, int, int], light_color: Tuple[int, int, int], n: float) -> Tuple[int, int, int, float]:
    """
    颜色插值，返回 (r,g,b,a)；n∈[0,1]
    """
    n = max(0.0, min(1.0, float(n)))
    r = int(dark_color[0] + (light_color[0] - dark_color[0]) * n)
    g = int(dark_color[1] + (light_color[1] - dark_color[1]) * n)
    b = int(dark_color[2] + (light_color[2] - dark_color[2]) * n)
    return r, g, b, 0.7


# ---------------------------
# Subway（NYCT）
# ---------------------------
def get_subway_schedule() -> List[Dict]:
    """
    汇总所有 NYCT 子 feed 的 trip_update → list[dict]
    dict keys: route, arrival_time, departure_time, stop_id
    """
    api_key = _get_subway_key()
    headers = {"x-api-key": api_key} if api_key else {}

    urls = [
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-ace",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-bdfm",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-g",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-jz",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-l",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-nqrw",
        "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs-si",
    ]

    feed = gtfs_realtime_pb2.FeedMessage()
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
            resp.raise_for_status()
            fm = gtfs_realtime_pb2.FeedMessage()
            fm.ParseFromString(resp.content)
            feed.entity.extend(fm.entity)
        except Exception:
            # 单个子 feed 出错不影响其他
            continue

    rows: List[Dict] = []
    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            route_id = tu.trip.route_id if tu.trip.HasField("route_id") else ""
            for stu in tu.stop_time_update:
                _append_stop_time(rows, route_id, stu)

    return rows


# ---------------------------
# Metro-North Railroad（MNR）
# ---------------------------
def get_MNR_schedule() -> List[Dict]:
    api_key = _get_subway_key()
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/mnr%2Fgtfs-mnr"

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        fm = gtfs_realtime_pb2.FeedMessage()
        fm.ParseFromString(resp.content)
        feed.entity.extend(fm.entity)
    except Exception:
        return []

    rows: List[Dict] = []
    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            route_id = tu.trip.route_id if tu.trip.HasField("route_id") else ""
            for stu in tu.stop_time_update:
                _append_stop_time(rows, route_id, stu)
    return rows


# ---------------------------
# Long Island Rail Road（LIRR）
# ---------------------------
def get_LIRR_schedule() -> List[Dict]:
    api_key = _get_subway_key()
    headers = {"x-api-key": api_key} if api_key else {}
    url = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/lirr%2Fgtfs-lirr"

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        fm = gtfs_realtime_pb2.FeedMessage()
        fm.ParseFromString(resp.content)
        feed.entity.extend(fm.entity)
    except Exception:
        return []

    rows: List[Dict] = []
    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            route_id = tu.trip.route_id if tu.trip.HasField("route_id") else ""
            for stu in tu.stop_time_update:
                _append_stop_time(rows, route_id, stu)
    return rows


# ---------------------------
# NYC Bus（OBANYC）
# ---------------------------
def get_bus_schedule() -> List[Dict]:
    """
    OBANYC tripUpdates → list[dict]
    """
    key = _get_bus_key()
    base_url = "http://gtfsrt.prod.obanyc.com/tripUpdates"
    request_url = f"{base_url}?key={key}" if key else base_url

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        resp = requests.get(request_url, timeout=TIMEOUT)
        resp.raise_for_status()
        feed.ParseFromString(resp.content)
    except Exception:
        return []

    rows: List[Dict] = []
    for entity in feed.entity:
        if entity.HasField("trip_update"):
            tu = entity.trip_update
            route_id = tu.trip.route_id if tu.trip.HasField("route_id") else ""
            for stu in tu.stop_time_update:
                _append_stop_time(rows, route_id, stu)
    return rows


def get_bus_location() -> List[Dict]:
    """
    OBANYC vehiclePositions → list[dict]
    """
    key = _get_bus_key()
    base_url = "http://gtfsrt.prod.obanyc.com/vehiclePositions"
    request_url = f"{base_url}?key={key}" if key else base_url

    feed = gtfs_realtime_pb2.FeedMessage()
    try:
        resp = requests.get(request_url, timeout=TIMEOUT)
        resp.raise_for_status()
        feed.ParseFromString(resp.content)
    except Exception:
        return []

    rows: List[Dict] = []
    for entity in feed.entity:
        if entity.HasField("vehicle"):
            v = entity.vehicle
            rows.append(
                {
                    "Vehicle ID": getattr(v.vehicle, "id", None),
                    "Route ID": v.trip.route_id if v.trip.HasField("route_id") else "",
                    "Direction ID": v.trip.direction_id if v.trip.HasField("direction_id") else None,
                    "Latitude": getattr(v.position, "latitude", None),
                    "Longitude": getattr(v.position, "longitude", None),
                }
            )
    return rows
