import requests
from google.transit import gtfs_realtime_pb2
from datetime import datetime

with open("GTFS/subway_API_Key.txt", "r") as f:
    subway_API_KEY = f.read().strip()

with open("GTFS/bus_API_Key.txt", "r") as f:
    bus_API_KEY = f.read().strip()


def get_subway_schedule():
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
        response = requests.get(url, headers={"x-api-key": subway_API_KEY})
        feed_message = gtfs_realtime_pb2.FeedMessage()
        feed_message.ParseFromString(response.content)
        feed.entity.extend(feed_message.entity)

    subway_schedule = []

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            trip_update = entity.trip_update
            for stop_time_update in trip_update.stop_time_update:
                arrival_time = datetime.fromtimestamp(
                    stop_time_update.arrival.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                departure_time = datetime.fromtimestamp(
                    stop_time_update.departure.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                stop_id = stop_time_update.stop_id
                subway_schedule.append(
                    {
                        "route": trip_update.trip.route_id,
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "stop_id": stop_id,
                    }
                )

    return subway_schedule


def get_MNR_schedule():
    url = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/mnr%2Fgtfs-mnr"
    feed = gtfs_realtime_pb2.FeedMessage()
    response = requests.get(url, headers={"x-api-key": subway_API_KEY})
    feed_message = gtfs_realtime_pb2.FeedMessage()
    feed_message.ParseFromString(response.content)
    feed.entity.extend(feed_message.entity)

    MNR_schedule = []

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            trip_update = entity.trip_update
            for stop_time_update in trip_update.stop_time_update:
                arrival_time = datetime.fromtimestamp(
                    stop_time_update.arrival.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                departure_time = datetime.fromtimestamp(
                    stop_time_update.departure.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                stop_id = stop_time_update.stop_id
                MNR_schedule.append(
                    {
                        "route": trip_update.trip.route_id,
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "stop_id": stop_id,
                    }
                )

    return MNR_schedule


def get_LIRR_schedule():
    url = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/lirr%2Fgtfs-lirr"
    feed = gtfs_realtime_pb2.FeedMessage()
    response = requests.get(url, headers={"x-api-key": subway_API_KEY})
    feed_message = gtfs_realtime_pb2.FeedMessage()
    feed_message.ParseFromString(response.content)
    feed.entity.extend(feed_message.entity)

    LIRR_schedule = []

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            trip_update = entity.trip_update
            for stop_time_update in trip_update.stop_time_update:
                arrival_time = datetime.fromtimestamp(
                    stop_time_update.arrival.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                departure_time = datetime.fromtimestamp(
                    stop_time_update.departure.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                stop_id = stop_time_update.stop_id
                LIRR_schedule.append(
                    {
                        "route": trip_update.trip.route_id,
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "stop_id": stop_id,
                    }
                )

    return LIRR_schedule


def get_bus_schedule():
    base_url = "http://gtfsrt.prod.obanyc.com/tripUpdates"
    request_url = f"{base_url}?key={bus_API_KEY}"

    response = requests.get(request_url)
    data = response.content

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    bus_schedule = []

    for entity in feed.entity:
        if entity.HasField("trip_update"):
            trip_update = entity.trip_update
            route_id = trip_update.trip.route_id
            for stop_time_update in trip_update.stop_time_update:
                arrival_time = datetime.fromtimestamp(
                    stop_time_update.arrival.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                departure_time = datetime.fromtimestamp(
                    stop_time_update.departure.time
                ).strftime("%Y-%m-%d %H:%M:%S")
                stop_id = stop_time_update.stop_id
                bus_schedule.append(
                    {
                        "route": route_id,
                        "arrival_time": arrival_time,
                        "departure_time": departure_time,
                        "stop_id": stop_id,
                    }
                )

    return bus_schedule


def get_bus_location():
    base_url = "http://gtfsrt.prod.obanyc.com/vehiclePositions"
    request_url = f"{base_url}?key={bus_API_KEY}"

    response = requests.get(request_url)
    data = response.content

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(data)

    location = []

    for entity in feed.entity:
        if entity.HasField("vehicle"):
            vehicle = entity.vehicle
            vehicle_id = vehicle.vehicle.id
            route_id = (
                vehicle.trip.route_id if vehicle.trip.HasField("route_id") else ""
            )
            direction_id = (
                vehicle.trip.direction_id
                if vehicle.trip.HasField("direction_id")
                else None
            )
            latitude = vehicle.position.latitude
            longitude = vehicle.position.longitude
            location.append(
                {
                    "Vehicle ID": vehicle_id,
                    "Route ID": route_id,
                    "Direction ID": direction_id,
                    "Latitude": latitude,
                    "Longitude": longitude,
                }
            )

    return location


def color_interpolation(dark_color: tuple, light_color: tuple, n: float) -> tuple:
    r = int(dark_color[0] + (light_color[0] - dark_color[0]) * n)
    g = int(dark_color[1] + (light_color[1] - dark_color[1]) * n)
    b = int(dark_color[2] + (light_color[2] - dark_color[2]) * n)
    return r, g, b, 0.7
