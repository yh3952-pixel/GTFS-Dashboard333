import os
from utils import (
    get_bus_schedule,
    get_subway_schedule,
    get_LIRR_schedule,
    get_MNR_schedule,
    color_interpolation,
)
from datetime import datetime
import json
import urllib
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

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
    "NJ_rail",
]
dataframes = {}
for subdir in SUBFILES:
    folder_path = os.path.join("GTFS", subdir)

    routes = pd.read_csv(os.path.join(folder_path, "routes.txt"))
    stop_times = pd.read_csv(os.path.join(folder_path, "stop_times.txt"))
    stops = pd.read_csv(os.path.join(folder_path, "stops.txt"))
    trips = pd.read_csv(os.path.join(folder_path, "trips.txt"))

    df = trips[["route_id", "service_id", "trip_id"]]
    df = df.merge(
        stop_times[
            ["trip_id", "arrival_time", "departure_time", "stop_sequence", "stop_id"]
        ],
        left_on="trip_id",
        right_on="trip_id",
        how="left",
    )
    df = df.merge(
        stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
        left_on="stop_id",
        right_on="stop_id",
        how="left",
    )
    df = df.merge(
        routes[["route_id", "route_long_name", "route_color"]],
        left_on="route_id",
        right_on="route_id",
        how="left",
    )

    route_color_mapping = (
        df.set_index("route_id")["route_color"]
        .fillna("000000")
        .astype(str)
        .apply(lambda x: "#" + x)
        .to_dict()
    )
    df["color"] = df["route_id"].map(route_color_mapping)
    dataframes[subdir] = df

dataframes["bus_new_jersy"]["color"] = "#00FF00"
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten_Island", "New_Jersy"]
CITIBIKE_REGIONS = ["NYC District", "JC District", "Hoboken District"]
SUBWAY_ID = dataframes["subway"]["route_id"].unique()
BUS_BRONX_ROUTE = dataframes["bus_bronx"]["route_id"].unique()
BUS_BROOKLYN_ROUTE = dataframes["bus_brooklyn"]["route_id"].unique()
BUS_MANHATTAN_ROUTE = dataframes["bus_manhattan"]["route_id"].unique()
BUS_QUEENS_ROUTE = dataframes["bus_queens"]["route_id"].unique()
BUS_STATEN_ISLAND_ROUTE = dataframes["bus_staten_island"]["route_id"].unique()
BUS_NEW_JERSY_ROUTE = dataframes["bus_new_jersy"]["route_id"].unique()
BUS_LIRR_ROUTE = dataframes["LIRR"]["route_id"].unique()
MNR_ROUTE = dataframes["MNR"]["route_id"].unique()
NJ_RAIL_ROUTE = dataframes["NJ_rail"]["route_id"].unique()
BUS_ROUTE_MAPPING = {
    "Bronx": BUS_BRONX_ROUTE,
    "Brooklyn": BUS_BROOKLYN_ROUTE,
    "Manhattan": BUS_MANHATTAN_ROUTE,
    "Queens": BUS_QUEENS_ROUTE,
    "Staten_Island": BUS_STATEN_ISLAND_ROUTE,
    "New_Jersy": BUS_NEW_JERSY_ROUTE,
}

BOROUGHS_COORDINATE_MAPPING = {
    "Bronx": [40.837048, -73.865433],
    "Brooklyn": [40.650002, -73.949997],
    "Manhattan": [40.776676, -73.971321],
    "Queens": [40.742054, -73.769417],
    "Staten_Island": [40.579021, -74.151535],
    "New_Jersy": [39.833851, -74.871826],
}

MAP_DIR_MAPPING = {
    "subway": os.path.join("assets", "subway_mapping", "subway.html"),
    "LIRR": os.path.join("assets", "LIRR_mapping", "LIRR.html"),
    "bus_bronx": os.path.join("assets", "bus_mapping", "bus_bronx.html"),
    "bus_brooklyn": os.path.join("assets", "bus_mapping", "bus_brooklyn.html"),
    "bus_mahattan": os.path.join("assets", "bus_mapping", "bus_manhattan.html"),
    "bus_queens": os.path.join("assets", "bus_mapping", "bus_queens.html"),
    "bus_staten_island": os.path.join(
        "assets", "bus_mapping", "bus_staten_island.html"
    ),
    "bus_new_jersy": os.path.join("assets", "bus_mapping", "bus_new_jersy.html"),
    "citibike": os.path.join("assets", "citibike_mapping", "citibike.html"),
}

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
# styles = {'background': '#262729', 'textColor': '#ffffff', 'marginColor': '#0e1012'}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server
app.layout = html.Div(
    style={
        "min-height": "100vh",
        "margin": "0 auto",
        "padding": "0 20px",
        # "max-width": "1200px",
        "box-sizing": "border-box",
    },
    children=[
        html.Br(),
        html.Br(),
        html.H1(
            children="Real Time Transportation Dashboard",
            style={"textAlign": "center", "font-weight": "bold"},
        ),
        html.Div(
            style={"display": "flex", "justify-content": "flex-end"},
            children=[
                html.Button(
                    "Unselect all routes",
                    id="unselect_btn",
                    style={"margin": "1.25%", "max-height": "50px"},
                    n_clicks=0,
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "height": "90vh"},
            children=[
                dcc.Interval(id="refresh_interval", interval=120 * 1000),
                html.Div(
                    style={
                        "flex": "0 0 21%",
                        "margin": "1%",
                        "boxSizing": "border-box",
                    },
                    children=[
                        html.H2("Subway"),
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        html.Button(
                                            "All routes",
                                            id="subway_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.H2("Citibike"),
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        html.Button(
                                            "Citibike Map",
                                            id="citibike_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.H2("LIRR"),
                        html.Div(
                            children=[
                                html.Div(
                                    [
                                        html.Button(
                                            "All routes",
                                            id="LIRR_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        html.H2("Bus"),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10 px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Button(
                                            "Bronx",
                                            id="bus_bronx_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "Brooklyn",
                                            id="bus_brooklyn_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "Manhattan",
                                            id="bus_mahattan_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "Queens",
                                            id="bus_queens_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "Staten_Island",
                                            id="bus_staten_island_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "New_Jersy",
                                            id="bus_new_jersy_btn",
                                            style={"margin": "1.25%"},
                                            n_clicks=0,
                                        ),
                                        dcc.Store(
                                            id="button_store",
                                            data=os.path.join(
                                                "assets", "default_map.html"
                                            ),
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.Div(
                            style={"padding": "10px", "boxSizing": "border-box"},
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                html.Iframe(
                                                    src=os.path.join(
                                                        "assets", f"default_map.html"
                                                    ),
                                                    id="real_time_map",
                                                    style={
                                                        "height": "1000px",
                                                        "width": "100%",
                                                        "border": "none",
                                                    },
                                                ),
                                                dcc.Interval(
                                                    id="interval-component",
                                                    interval=60 * 1000,
                                                    n_intervals=0,
                                                ),
                                            ]
                                        ),
                                    ],
                                    style={
                                        "border": "none",
                                        "background-color": "#f8f9fa",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
app.clientside_callback(
    """
        function(n_clicks) {
            if (n_clicks == 0) {
                return
            }
            const iframe = document.querySelector("#real_time_map")
            const iframe_doc = iframe.contentDocument || iframe.contentWindow.document
            const graph = iframe_doc.getElementsByClassName("plotly-graph-div")[0]
            if (graph) {
                iframe_doc.defaultView.Plotly.restyle(graph, {visible: 'legendonly'})
                .then(() => {console.log("All routes unselected.")})
                .catch((e) => {console.error(e)});
            }
        }
    """,
    Input("unselect_btn", "n_clicks"),
)


def filter_feed_df(schedule_feed_df: pd.DataFrame) -> pd.DataFrame:
    # if arrival time is invalid fill with departure time
    mask = (schedule_feed_df["arrival_time"].isna()) | (
        pd.to_datetime(schedule_feed_df["arrival_time"])
        < datetime.now().replace(hour=0, minute=0, second=0)
    )
    schedule_feed_df.loc[mask, "arrival_time"] = schedule_feed_df.loc[
        mask, "departure_time"
    ]
    mask = (schedule_feed_df["departure_time"].isna()) | (
        pd.to_datetime(schedule_feed_df["departure_time"])
        < datetime.now().replace(hour=0, minute=0, second=0)
    )
    # drop invalid rows
    schedule_feed_df.loc[mask, "departure_time"] = schedule_feed_df.loc[
        mask, "arrival_time"
    ]
    schedule_feed_df = schedule_feed_df[~mask]

    # find the min arrival time for each (route, stop) pair
    schedule_feed_df = (
        schedule_feed_df.groupby(["route", "stop_id"])[
            ["arrival_time", "departure_time"]
        ]
        .min()
        .reset_index()
    )
    return schedule_feed_df


def init_bus_map(schedule_feed_df: pd.DataFrame) -> dict:
    bus_trace_dict = {}
    schedule_feed_df["stop_id"] = schedule_feed_df["stop_id"].astype("int")
    for borough in BOROUGHS:
        bus_borough_traces = []
        borough_bus_df = dataframes[f"bus_{borough.lower()}"][
            [
                "route_id",
                "stop_id",
                "stop_lat",
                "stop_lon",
                "stop_sequence",
                "color",
                "stop_name",
            ]
        ].drop_duplicates(subset=["route_id", "stop_id", "stop_sequence"])
        for route in borough_bus_df["route_id"].unique():
            route_bus_df = borough_bus_df[borough_bus_df["route_id"] == route].copy()
            # using sequence value to find the next stop position
            route_bus_df.loc[:, "next_sequence"] = route_bus_df["stop_sequence"].shift(
                -1
            )
            route_bus_df.loc[
                route_bus_df["next_sequence"] != route_bus_df["stop_sequence"] + 1,
                "next_sequence",
            ] = -1
            route_bus_df = route_bus_df.drop_duplicates(
                subset=["next_sequence", "stop_id"]
            ).reset_index()
            route_bus_schedule_df = schedule_feed_df[schedule_feed_df["route"] == route]
            route_bus_df["arrival_time"] = route_bus_df.merge(
                route_bus_schedule_df,
                how="left",
                left_on=["stop_id"],
                right_on=["stop_id"],
            )["arrival_time"]
            route_bus_df["arrival_time"] = route_bus_df["arrival_time"].fillna("N/A")
            subroute_idx = route_bus_df[
                (~route_bus_df["next_sequence"].isna())
                & (
                    (route_bus_df["next_sequence"] == -1)
                    | (
                        route_bus_df["next_sequence"]
                        != route_bus_df["stop_sequence"].shift(-1)
                    )
                )
            ].index
            if len(subroute_idx) == 0:
                subroutes = [route_bus_df]
            else:
                subroutes = [
                    (
                        route_bus_df[i + 1 : j + 1]
                        if i + 1 != j + 1
                        else route_bus_df.iloc[i + 1]
                    )
                    for i, j in zip([-1] + list(subroute_idx), subroute_idx)
                ]
            color = route_bus_df["color"].iloc[0]
            if color == "#000000":
                color = "blue"
            bus_borough_traces.extend(
                [
                    go.Scattermapbox(
                        lon=subroute_df["stop_lon"],
                        lat=subroute_df["stop_lat"],
                        mode="markers+lines",
                        marker=dict(symbol="circle", color="white", size=4),
                        text=subroute_df.apply(
                            lambda x: f"""Route: {route} <br> Stop Name: {x['stop_name']} 
                                        <br> Next Arrival Time: {x['arrival_time']}""",
                            axis=1,
                        ),
                        hoverinfo="text",
                        line=dict(width=3, color=color),
                        legendgroup=f"bus_{route}",
                        legendgrouptitle={"text": f"bus route: {route}"},
                        name=f"{subroute_df['stop_name'].iloc[0]} to {subroute_df['stop_name'].iloc[-1]}",
                    )
                    for subroute_df in subroutes
                    if len(subroute_df) != 0
                ]
            )
        bus_trace_dict[f"{borough}"] = bus_borough_traces
    return bus_trace_dict


def init_subway_map(schedule_feed_df: pd.DataFrame) -> dict:
    subway_trace_dict = {}
    subway_df = dataframes["subway"]
    for route in SUBWAY_ID:
        subway_route_df = subway_df.loc[subway_df["route_id"] == route]
        subway_route_df = subway_route_df.drop_duplicates(subset=["stop_id"])[
            [
                "route_id",
                "stop_sequence",
                "stop_id",
                "stop_lat",
                "stop_lon",
                "route_long_name",
                "color",
                "stop_name",
            ]
        ]
        subway_route_df = subway_route_df.merge(
            schedule_feed_df,
            how="left",
            left_on=["route_id", "stop_id"],
            right_on=["route", "stop_id"],
        )
        subway_route_df["next_stop_sequence"] = subway_route_df.shift(-1)[
            "stop_sequence"
        ]
        subway_subroute_idx = subway_route_df[
            (~subway_route_df["next_stop_sequence"].isna())
            & (
                subway_route_df["next_stop_sequence"]
                != subway_route_df["stop_sequence"] + 1
            )
        ].index
        if len(subway_subroute_idx) == 0:
            subway_subroute = [subway_route_df]
        else:
            subway_subroute = [
                (
                    subway_route_df[i + 1 : j + 1]
                    if i + 1 != j + 1
                    else subway_route_df.iloc[i + 1]
                )
                for (i, j) in zip([-1] + list(subway_subroute_idx), subway_subroute_idx)
            ]
        color = subway_route_df["color"].iloc[0]
        if color == "#000000":
            color = "blue"
        subway_route_traces = [
            go.Scattermapbox(
                lon=subroute_df["stop_lon"],
                lat=subroute_df["stop_lat"],
                mode="markers+lines",
                marker=dict(symbol="circle", color="white", size=4),
                text=subroute_df.apply(
                    lambda x: f"""Route: {x['route_long_name']} <br> Stop Name: {x['stop_name']} 
                                <br> Next Arrival Time: {x['arrival_time']} 
                                <br> Next Departure Time: {x['departure_time']}""",
                    axis=1,
                ),
                hoverinfo="text",
                line=dict(width=3, color=color),
                legendgroup=f"subway_{route}",
                legendgrouptitle={"text": f"subway route: {route}"},
                name=f"{subroute_df['stop_name'].iloc[0]} to {subroute_df['stop_name'].iloc[-1]}",
            )
            for subroute_df in subway_subroute
            if len(subroute_df) != 0
        ]
        subway_trace_dict[route] = subway_route_traces
    return subway_trace_dict


def init_LIRR_map(schedule_feed_df: pd.DataFrame) -> dict:
    LIRR_trace_dict = {}
    LIRR_df = dataframes["LIRR"]

    # 类型统一（若不存在这些列，astype 会报错；若你确信存在可保留）
    schedule_feed_df[["stop_id"]] = schedule_feed_df[["stop_id"]].astype("int64")
    schedule_feed_df[["route"]]   = schedule_feed_df[["route"]].astype("int64")

    # 小工具：安全取颜色
    def _safe_color(df, default="#2E86DE"):
        if df is None or df.empty:
            return default
        col = "route_color" if "route_color" in df.columns else ("color" if "color" in df.columns else None)
        if not col:
            return default
        val = str(df[col].iloc[0]).strip()
        if not val or val.lower() == "nan":
            return default
        # GTFS 常见是 6 位 HEX 无 '#'
        if not val.startswith("#"):
            val = "#" + val
        # 特殊值黑色时换成蓝色（你原逻辑）
        if val.lower() in ("#000000", "#000"):
            return "blue"
        return val

    for route in BUS_LIRR_ROUTE:
        # 1) 先取该 route 的 LIRR 静态数据
        LIRR_route_df = LIRR_df.loc[LIRR_df["route_id"] == route].copy()

        # 判空：该 route 没静态点位就跳过
        if LIRR_route_df is None or LIRR_route_df.empty:
            # 保持键存在但为空列表，避免下游 KeyError
            LIRR_trace_dict[route] = []
            continue

        # 衔接上一站/序号断裂判断
        LIRR_route_df["last_stop_id"] = LIRR_route_df.shift(+1)["stop_id"]
        LIRR_route_df["last_stop_sequence"] = LIRR_route_df.shift(+1)["stop_sequence"]
        LIRR_route_df.loc[
            (
                (LIRR_route_df["stop_sequence"] == 1)
                | (
                    LIRR_route_df["last_stop_sequence"]
                    != LIRR_route_df["stop_sequence"] - 1
                )
                | pd.isna(LIRR_route_df["last_stop_sequence"])
            ),
            "last_stop_id",
        ] = -1
        LIRR_route_df["last_stop_id"] = (
            LIRR_route_df["last_stop_id"].fillna(-1).astype("int64")
        )

        # 去重并只保留必要列（兼容 route_color 缺失）
        keep_cols = [
            "route_id", "stop_sequence", "stop_id", "stop_lat", "stop_lon",
            "route_long_name", "stop_name", "last_stop_id"
        ]
        if "color" in LIRR_route_df.columns:
            keep_cols.append("color")
        if "route_color" in LIRR_route_df.columns and "route_color" not in keep_cols:
            keep_cols.append("route_color")

        LIRR_route_df = LIRR_route_df.drop_duplicates(subset=["stop_id"])[keep_cols]

        # 2) 与实时/时刻信息 merge
        merged = LIRR_route_df.merge(
            schedule_feed_df,
            left_on=["stop_id", "route_id"],
            right_on=["stop_id", "route"],
            how="inner",
        )

        # 判空：该 route 在当前时刻没班次也跳过
        if merged is None or merged.empty:
            LIRR_trace_dict[route] = []
            continue

        merged["next_stop_sequence"] = merged.shift(-1)["stop_sequence"]
        LIRR_subroute_idx = merged[
            (~merged["next_stop_sequence"].isna())
            & (merged["next_stop_sequence"] != merged["stop_sequence"] + 1)
        ].index

        if len(LIRR_subroute_idx) == 0:
            LIRR_subroute = [merged]
        else:
            LIRR_subroute = [
                (
                    merged[i + 1 : j + 1].reset_index(drop=True)
                    if i + 1 != j + 1
                    else merged.iloc[i + 1 : j + 1].reset_index(drop=True)
                )
                for (i, j) in zip([-1] + list(LIRR_subroute_idx), LIRR_subroute_idx)
            ]
            for i, subroute in enumerate(LIRR_subroute):
                if len(subroute) > 0 and subroute.loc[0, "last_stop_id"] != -1:
                    LIRR_subroute[i] = pd.concat(
                        [
                            merged.loc[merged["stop_id"] == subroute.loc[0, "last_stop_id"]],
                            subroute,
                        ],
                        ignore_index=True,
                    )

        # 颜色（安全）
        color = _safe_color(merged)

        # 构造图层（仍用 Scattermapbox，先跑通）
        subway_route_traces = [
            go.Scattermapbox(
                lon=subroute_df["stop_lon"],
                lat=subroute_df["stop_lat"],
                mode="markers+lines",
                marker=dict(symbol="circle", color="white", size=4),
                text=subroute_df.apply(
                    lambda x: (
                        f"Route: {x.get('route_long_name','')} <br>"
                        f"Stop Name: {x.get('stop_name','')} <br>"
                        f"Next Arrival Time: {x.get('arrival_time','')} <br>"
                        f"Next Departure Time: {x.get('departure_time','')}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
                line=dict(width=3, color=color),
                legendgroup=f"subway_{route}",
                legendgrouptitle={"text": f"subway route: {route}"},
                name=(
                    f"{subroute_df['stop_name'].iloc[0]} to {subroute_df['stop_name'].iloc[-1]}"
                    if len(subroute_df) > 0 else f"subway route: {route}"
                ),
            )
            for subroute_df in LIRR_subroute
            if subroute_df is not None and len(subroute_df) != 0
        ]

        LIRR_trace_dict[route] = subway_route_traces

    return LIRR_trace_dict


def citibike_station_data():
    station_info_url = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
    info_response = urllib.request.urlopen(station_info_url)
    data_info = json.load(info_response)

    station_status_url = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
    status_response = urllib.request.urlopen(station_status_url)
    data_status = json.load(status_response)

    system_regions_url = "https://gbfs.citibikenyc.com/gbfs/en/system_regions.json"
    regions_response = urllib.request.urlopen(system_regions_url)
    data_regions = json.load(regions_response)

    info_df = pd.DataFrame(data_info["data"]["stations"]).set_index("station_id")
    info_df = info_df[["name", "lat", "lon", "capacity", "region_id"]]

    status_df = pd.DataFrame(data_status["data"]["stations"]).set_index("station_id")
    status_df = status_df[
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

    regions_df = pd.DataFrame(data_regions["data"]["regions"])
    regions_df.rename(columns={"name": "region_name"}, inplace=True)

    citibike_df = info_df.merge(status_df, left_index=True, right_index=True)
    citibike_df = citibike_df.merge(
        regions_df[["region_id", "region_name"]],
        left_on="region_id",
        right_on="region_id",
    )
    return citibike_df


def init_citibike_map() -> dict:
    citibike_trace_dict = {}
    citibike_df = citibike_station_data()
    assert type(citibike_df) is pd.DataFrame
    citibike_df = citibike_df.sort_values(
        by=["lat", "lon", "last_reported"], ascending=False
    ).drop_duplicates(["lat", "lon"])[
        [
            "name",
            "lat",
            "lon",
            "region_name",
            "is_renting",
            "num_docks_available",
            "num_bikes_available",
            "num_ebikes_available",
            "last_reported",
        ]
    ]

    for region in CITIBIKE_REGIONS:
        citibike_region_df = citibike_df.loc[
            citibike_df["region_name"] == region
        ].copy()
        dark_color = CITIBIKE_REGIONS_COLORING_DARK[region]
        light_color = CITIBIKE_REGIONS_COLORING_LIGHT[region]
        citibike_region_df["color"] = citibike_region_df.apply(
            lambda x: f"rgba{color_interpolation(
                dark_color,
                light_color,
                min(x["num_bikes_available"], 80) / 80,
            )}",
            axis=1,
        )
        citibike_region_df["last_reported"] = citibike_region_df["last_reported"].apply(
            lambda x: datetime.fromtimestamp(int(x))
        )
        region_trace = go.Scattermapbox(
            lon=citibike_region_df["lon"],
            lat=citibike_region_df["lat"],
            mode="markers",
            marker=dict(
                size=13,
                color=citibike_region_df["color"],
            ),
            text=citibike_region_df.apply(
                lambda x: f"""Name: {x['name']} <br> Available Docks: {x['num_docks_available']} 
                <br> Available eBikes: {x['num_ebikes_available']} <br> Available Bikes: {x['num_bikes_available']}
                <br> Last Reported: {x['last_reported']}""",
                axis=1,
            ),
            legendgroup=f"citibike_{region}",
            legendgrouptitle={"text": region},
            name=f"citibike in {region}",
        )
        citibike_trace_dict[region] = region_trace
    return citibike_trace_dict


def init_MNR_map(schedule_feed_df: pd.DataFrame) -> dict:
    MNR_trace_dict = {}
    MNR_df = dataframes["MNR"]
    # for route in MNR_ROUTE:
    #     MNR_route_df = MNR_df.loc[MNR_df["route_id"] == route].copy()
    #     MNR_route_df = MNR_route_df.drop_duplicates(["stop_id"])
    #     MNR_route_df.merge(schedule_feed_df, on=["route", ""])
    

def init_NJrail_map() -> dict:
    pass


def init_default_map() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        mapbox={
            "center": {"lat": 40.8, "lon": -74},
            "style": "carto-darkmatter",
            "zoom": 10,
        },
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
    )
    fig.add_trace(go.Scattermapbox(lon=[40.8], lat=[-74], mode="markers", opacity=0))
    pio.write_html(
        fig,
        os.path.join("assets", "default_map.html"),
        auto_open=False,
        include_plotlyjs="cdn",
        full_html=False,
    )


@app.callback(Input("refresh_interval", "n_intervals"))
def generate_gtfs_map(n) -> go.Figure:
    subway_schedule_feed_df = filter_feed_df(pd.DataFrame(get_subway_schedule()))
    bus_schedule_feed_df = filter_feed_df(pd.DataFrame(get_bus_schedule()))
    LIRR_schedule_feed_df = filter_feed_df(pd.DataFrame(get_LIRR_schedule()))
    MNR_schedule_feed_df = filter_feed_df(pd.DataFrame(get_MNR_schedule()))
    bus_traces = init_bus_map(bus_schedule_feed_df)
    subway_traces = init_subway_map(subway_schedule_feed_df)
    LIRR_traces = init_LIRR_map(LIRR_schedule_feed_df)
    MNR_traces = init_MNR_map(MNR_schedule_feed_df)
    citibike_traces = init_citibike_map()
    for borough in bus_traces.keys():
        fig = go.Figure()
        fig.update_layout(
            mapbox={
                "center": {
                    "lat": BOROUGHS_COORDINATE_MAPPING[borough][0],
                    "lon": BOROUGHS_COORDINATE_MAPPING[borough][1],
                },
                "style": "carto-darkmatter",
                "zoom": 10,
            },
            margin=dict(l=0, r=0, b=0, t=0),
            hovermode="closest",
        )
        fig.add_traces(bus_traces[borough])
        pio.write_html(
            fig,
            os.path.join("assets", "bus_mapping", f"bus_{borough.lower()}.html"),
            auto_open=False,
            include_plotlyjs="cdn",
            full_html=False,
        )

    fig = go.Figure()
    fig.update_layout(
        mapbox={
            "center": {"lat": 40.8, "lon": -74},
            "style": "carto-darkmatter",
            "zoom": 10,
        },
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
    )
    for route in subway_traces.keys():
        fig.add_traces(subway_traces[route])
    pio.write_html(
        fig,
        os.path.join("assets", "subway_mapping", f"subway.html"),
        auto_open=False,
        include_plotlyjs="cdn",
        full_html=False,
    )

    fig = go.Figure()
    fig.update_layout(
        mapbox={
            "center": {"lat": 40.8, "lon": -74},
            "style": "carto-darkmatter",
            "zoom": 10,
        },
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
    )
    for route in LIRR_traces.keys():
        fig.add_traces(LIRR_traces[route])
    pio.write_html(
        fig,
        os.path.join("assets", "LIRR_mapping", "LIRR.html"),
        auto_open=False,
        include_plotlyjs="cdn",
        full_html=False,
    )

    fig = go.Figure()
    fig.update_layout(
        mapbox={
            "center": {"lat": 40.776676, "lon": -73.971321},
            "style": "carto-darkmatter",
            "zoom": 11,
        },
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode="closest",
    )
    for region in citibike_traces:
        fig.add_traces(citibike_traces[region])
        pio.write_html(
            fig,
            os.path.join("assets", "citibike_mapping", "citibike.html"),
            auto_open=False,
            include_plotlyjs="cdn",
            full_html=False,
        )


@app.callback(
    Output("button_store", "data"),
    Input("subway_btn", "n_clicks"),
    Input("LIRR_btn", "n_clicks"),
    Input("citibike_btn", "n_clicks"),
    Input("bus_bronx_btn", "n_clicks"),
    Input("bus_brooklyn_btn", "n_clicks"),
    Input("bus_mahattan_btn", "n_clicks"),
    Input("bus_queens_btn", "n_clicks"),
    Input("bus_staten_island_btn", "n_clicks"),
    Input("bus_new_jersy_btn", "n_clicks"),
)
def handle_button_clicked(
    btn_1, btn_2, btn_3, btn_4, btn_5, btn_6, btn_7, btn_8, btn_9
):
    ctx = callback_context
    if not ctx.triggered:
        return None
    triggered_button = ctx.triggered[0]["prop_id"].split(".")[0].replace("_btn", "")

    return MAP_DIR_MAPPING[triggered_button]


@app.callback(
    Output("real_time_map", "src"),
    Input("button_store", "data"),
    Input("refresh_interval", "n_intervals"),
)
def refresh_map_file(map_dir, n_intervals) -> os.path:
    return map_dir


if __name__ == "__main__":
    init_default_map()
    generate_gtfs_map(0)
    app.run(host="0.0.0.0", port=8050, debug=False)
