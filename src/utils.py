import math
import os
import logging
from typing import Union, Dict, Any, List

from scipy.ndimage import gaussian_filter1d
import numpy as np
from shapely.geometry import Polygon, Point
from pyproj import Transformer, CRS

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__))))
LOG_LEVEL = "WARNING"
LOG_FILE = f"{parent_dir}/logs/golf-course-plotting.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger.info(f"root_dir: {root_dir}")
smooth_sigma = 1
dpi = 300
meters_per_pixel = 0.1
lat_to_meter_ratio = 111000
base_area = 10 * 10  # 假设10x10英寸为基准尺寸
marker_in_meters = 3
marker_icon_pixels = 300
input_crs = CRS.from_string("EPSG:4326")  # 经纬度坐标系
output_crs = CRS.from_string("EPSG:3857")  # 墨卡托坐标系（以米为单位）
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)


def smooth_coordinates(coords):
    longitudes, latitudes = zip(*coords)
    smooth_longs = gaussian_filter1d(longitudes, smooth_sigma)
    smooth_lats = gaussian_filter1d(latitudes, smooth_sigma)
    return list(zip(smooth_longs, smooth_lats))


def get_smooth_polygon(coords):
    count = len(coords)
    if count < 3:
        logger.warning(f"坐标点数量不足以创建多边形: {count} 个点")
        return None
    # 确保多边形是闭合的（首尾坐标相同）
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    smoothed_coords = smooth_coordinates(coords)
    try:
        return Polygon(smoothed_coords)
    except ValueError as e:
        logger.warning(f"创建多边形失败: {e}")
        return None


def inside_polygons(coord, polygons):
    point = Point(coord)
    for polygon in polygons:
        if polygon and polygon.contains(point):
            return True
    return False


def intersection_of_polygons(polygon1, polygons):
    try:
        if not polygon1 or not polygons:
            return None
        if not polygon1.is_valid:
            return None
        for polygon2 in polygons:
            if not polygon2.is_valid:
                return None
            intersection = polygon1.intersection(polygon2)
            if intersection and not intersection.is_empty:
                return intersection
        return None
    except Exception as e:
        logger.warning(f"计算多边形相交时出错: {e}")
        return None


def transform_coordinates(coords):
    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.ndim == 1:
        x, y = transformer.transform(coords[0], coords[1])
        return np.array([x, y])
    else:
        return np.array([transformer.transform(lon, lat) for lon, lat in coords])

def get_unique_ascending(arr):
    ret = list(set(arr))
    ret.sort()
    return ret, ret[0], ret[-1]

# Remove values from isolated points
def get_duplicated_values(values, arr):
    ret = []
    for v in values:
        count = 0
        for tmp in arr:
            if tmp == v:
                count += 1
        if count >= 3:
            ret.append(v)
    return ret


def is_same(pointA, pointB):
    return pointA[0] == pointB[0] and pointA[1] == pointB[1]

# get index of the nearest point on edge cycle from pnt
# edges[i] is in board index
def nearest_index(pnt, edges, xdup, ydup):
    for i in range(len(edges)):
        edge_point = [xdup[edges[i][0]], ydup[edges[i][1]]]
        d = math.dist(pnt, edge_point)
        if i == 0:
            result = 0
            min_dist = d
        else:
            if d < min_dist:
                min_dist = d
                result = i

    return result

# returns an array, starts from a, ends from b
# a, b are indices on a cycle of length n
# so select a small path from a to b
def get_indices(a, b, n):
    is_reverse = False
    if a > b:
        a, b = b, a
        is_reverse = True

    c = a + n

    if b - a < c - b:
        ret = [i for i in range(a, b + 1)]
    else:
        ret = [i % n for i in range(b, c + 1)]
        is_reverse = not is_reverse

    if is_reverse:
        return ret[::-1]
    else:
        return ret


def get_mid_point(pointA, pointB, ratio):
    pa = np.asarray(pointA)
    pb = np.asarray(pointB)
    return pa * (1 - ratio) + pb * ratio


def convert_json_num_to_str(json_data: Union[Dict[str, Any], List[Any], int, float, Any]) -> Union[Dict[str, Any], List[Any], str, Any]:
    if isinstance(json_data, dict):
        return {k: convert_json_num_to_str(v) for k, v in json_data.items()}
    elif isinstance(json_data, list):
        return [convert_json_num_to_str(item) for item in json_data]
    elif isinstance(json_data, (int, float)):
        return str(json_data)
    else:
        return json_data