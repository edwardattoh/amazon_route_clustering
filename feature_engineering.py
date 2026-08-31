import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def load_json(file_path: Path):
    """
    Load a JSON file from the computer.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}. "
            "Run download_data.py first."
        )

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value):
    """
    Try to convert a value into a number.
    Return a missing-value marker if conversion is not possible.
    """
    if value is None:
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def haversine_distance_km(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2,
):
    """
    Calculate the distance between two latitude and longitude positions.
    The result is returned in kilometres.
    """
    lat_1 = math.radians(latitude_1)
    lon_1 = math.radians(longitude_1)
    lat_2 = math.radians(latitude_2)
    lon_2 = math.radians(longitude_2)

    latitude_difference = lat_2 - lat_1
    longitude_difference = lon_2 - lon_1

    a = (
        math.sin(latitude_difference / 2) ** 2
        + math.cos(lat_1)
        * math.cos(lat_2)
        * math.sin(longitude_difference / 2) ** 2
    )

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def get_delivery_coordinates(route):
    """
    Collect valid latitude and longitude values from delivery stops.

    The station or depot is not counted as a delivery stop.
    """
    coordinates = []

    for stop in route.get("stops", {}).values():
        stop_type = str(stop.get("type", "")).strip().lower()

        if stop_type == "station":
            continue

        latitude = safe_float(stop.get("lat"))
        longitude = safe_float(stop.get("lng"))

        if np.isfinite(latitude) and np.isfinite(longitude):
            coordinates.append((latitude, longitude))

    return coordinates


def calculate_geographic_features(coordinates):
    """
    Calculate the route's average location and geographic spread.

    Geographic spread means the average distance from every delivery
    stop to the route's average location.
    """
    if not coordinates:
        return np.nan, np.nan, np.nan

    latitudes = [coordinate[0] for coordinate in coordinates]
    longitudes = [coordinate[1] for coordinate in coordinates]

    average_latitude = float(np.mean(latitudes))
    average_longitude = float(np.mean(longitudes))

    distances = [
        haversine_distance_km(
            latitude,
            longitude,
            average_latitude,
            average_longitude,
        )
        for latitude, longitude in coordinates
    ]

    geographic_spread = float(np.mean(distances))

    return (
        average_latitude,
        average_longitude,
        geographic_spread,
    )


def calculate_package_features(route_packages):
    """
    Calculate package information for one complete route.
    """
    package_count = 0
    total_service_time_seconds = 0.0
    package_volumes = []

    for stop_packages in route_packages.values():
        if not isinstance(stop_packages, dict):
            continue

        for package in stop_packages.values():
            if not isinstance(package, dict):
                continue

            package_count += 1

            service_time = safe_float(
                package.get("planned_service_time_seconds")
            )

            if np.isfinite(service_time):
                total_service_time_seconds += service_time

            dimensions = package.get("dimensions", {})

            depth = safe_float(dimensions.get("depth_cm"))
            height = safe_float(dimensions.get("height_cm"))
            width = safe_float(dimensions.get("width_cm"))

            if (
                np.isfinite(depth)
                and np.isfinite(height)
                and np.isfinite(width)
                and depth >= 0
                and height >= 0
                and width >= 0
            ):
                volume = depth * height * width
                package_volumes.append(volume)

    if package_volumes:
        average_package_volume = float(
            np.mean(package_volumes)
        )
    else:
        average_package_volume = np.nan

    return (
        package_count,
        total_service_time_seconds,
        average_package_volume,
    )


def build_route_feature_table(
    route_data,
    package_data,
    sample_size=500,
    random_seed=42,
):
    """
    Create one row for every sampled route.
    """
    common_route_ids = sorted(
        set(route_data.keys()).intersection(package_data.keys())
    )

    if not common_route_ids:
        raise ValueError(
            "No matching route IDs were found in the two data files."
        )

    if sample_size <= 0:
        raise ValueError(
            "The sample size must be greater than zero."
        )

    actual_sample_size = min(
        sample_size,
        len(common_route_ids),
    )

    random_generator = random.Random(random_seed)

    sampled_route_ids = random_generator.sample(
        common_route_ids,
        actual_sample_size,
    )

    rows = []

    for route_id in sampled_route_ids:
        route = route_data[route_id]
        route_packages = package_data[route_id]

        coordinates = get_delivery_coordinates(route)

        (
            average_latitude,
            average_longitude,
            geographic_spread,
        ) = calculate_geographic_features(coordinates)

        (
            package_count,
            total_service_time_seconds,
            average_package_volume,
        ) = calculate_package_features(route_packages)

        row = {
            "route_id": route_id,
            "number_of_stops": len(coordinates),
            "number_of_packages": package_count,
            "total_planned_service_time_seconds":
                total_service_time_seconds,
            "average_stop_latitude": average_latitude,
            "average_stop_longitude": average_longitude,
            "geographic_spread_km": geographic_spread,
            "average_package_volume_cm3":
                average_package_volume,
            "vehicle_capacity_cm3": safe_float(
                route.get("executor_capacity_cm3")
            ),
            "route_score": route.get("route_score"),
        }

        rows.append(row)

    feature_table = pd.DataFrame(rows)

    if feature_table.empty:
        raise ValueError(
            "The route feature table is empty."
        )

    return feature_table