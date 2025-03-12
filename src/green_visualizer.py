import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as colors

smooth_sigma = 2
colors_list = [
  '#000080',  # navy blue
  '#0000FF',  # blue
  '#00FF00',  # green
  '#80FF00',  # yellow-green
  '#FFFF00',  # yellow
  '#FFA500',  # orange
  '#FF8000',  # dark orange
  '#FF0000'   # red
]

def smooth_coordinates(coords):
    longitudes, latitudes = zip(*coords)
    smooth_longs = gaussian_filter1d(longitudes, smooth_sigma)
    smooth_lats = gaussian_filter1d(latitudes, smooth_sigma)
    return list(zip(smooth_longs, smooth_lats))


def get_smooth_polygon(coords):
    count = len(coords)
    if count < 3:
        print(f"not enough points to create polygon: {count} points")
        return None
    # ensure the polygon is closed
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    smoothed_coords = smooth_coordinates(coords)
    try:
        return Polygon(smoothed_coords)
    except ValueError as e:
        print(f"failed to create polygon: {e}")
        return None
     
def inside_polygon(coord, polygon):
    point = Point(coord)
    return polygon and polygon.contains(point)
       
class GreenVisualizer:
    def __init__(self, json_path, output_path):
        self.data = self._load_json(json_path)
        self.elevation_points = []
        self.green_border = None
        self.parse_data()
        self.output_path = output_path
    def _load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def parse_data(self):
        """Parse JSON data, extract elevation points and boundary"""
        for feature in self.data['features']:
            if feature['id'] == 'Elevation':
                coords = feature['geometry']['coordinates']
                self.elevation_points.append({
                    'x': coords[0],
                    'y': coords[1],
                    'z': coords[2]
                })
            elif feature['id'] == 'GreenBorder':
                coords = feature['geometry']['coordinates']
                self.green_border = Polygon(coords)

    def plot(self, resolution=100):
        """Create visualization of the green"""
        # Convert point data to numpy arrays
        points = np.array([[p['x'], p['y']] for p in self.elevation_points])
        values = np.array([p['z'] for p in self.elevation_points])
        
        # Create grid with consistent spacing
        x_min, x_max = min(points[:,0]), max(points[:,0])
        y_min, y_max = min(points[:,1]), max(points[:,1])
        
        # Calculate the actual physical distances
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Create grid points with proper aspect ratio
        x_resolution = int(resolution * (x_range / max(x_range, y_range)))
        y_resolution = int(resolution * (y_range / max(x_range, y_range)))
        
        xi = np.linspace(x_min, x_max, x_resolution)
        yi = np.linspace(y_min, y_max, y_resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolation
        zi = griddata(points, values, (xi, yi), method='cubic')
        
        # Build smooth boundary from points
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        edge_points = points[hull.vertices]
        smoothed_edge_points = smooth_coordinates(edge_points)
        smooth_border = get_smooth_polygon(smoothed_edge_points)
        
        # Create mask using smooth border
        mask = np.zeros_like(xi, dtype=bool)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                point = Point(xi[i,j], yi[i,j])
                mask[i,j] = smooth_border.contains(point) if smooth_border else True
        
        # Apply mask to interpolated values
        zi_masked = np.ma.masked_array(zi, ~mask)
        
        # Create plot with proper aspect ratio
        aspect_ratio = y_range / x_range
        fig_width = 10
        fig_height = fig_width * aspect_ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # 首先绘制颜色渐变
        levels = np.linspace(values.min(), values.max(), 20)
        custom_cmap = colors.LinearSegmentedColormap.from_list('custom', colors_list)
        contour = ax.contourf(xi, yi, zi_masked, levels=levels, cmap=custom_cmap)
        
        # 然后绘制等高线
        ax.contour(xi, yi, zi_masked, levels=levels, colors='k', alpha=0.3)
        
        # 计算和打印梯度信息
        dx, dy = np.gradient(zi_masked)
        print("Gradient statistics:")
        print(f"dx range: {np.nanmin(dx)} to {np.nanmax(dx)}")
        print(f"dy range: {np.nanmin(dy)} to {np.nanmax(dy)}")
        
        # 将所有箭头标准化为单位向量
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitude = np.where(magnitude == 0, 1, magnitude)
        dx_normalized = dx / magnitude
        dy_normalized = dy / magnitude
        
        skip = (slice(None, None, 8), slice(None, None, 8))
        mask_skip = mask[skip]
        
        ax.quiver(xi[skip][mask_skip], yi[skip][mask_skip], 
                 -dx_normalized[skip][mask_skip], -dy_normalized[skip][mask_skip], 
                 scale=10,          # 调整scale使箭头大小合适
                 scale_units='width',
                 units='width',
                 width=0.05,       # 箭头线的粗细
                 headwidth=15,       # 箭头头部的宽度
                 headlength=15,      # 箭头头部的长度
                 headaxislength=10, # 箭头头部底部的长度
                 minshaft=10,        # 增加最小轴长，确保箭头有足够长的尾部
                 minlength=5,     # 增加最小总长度
                 color='white', 
                 alpha=0.8)
        
        plt.savefig(self.output_path, 
                bbox_inches='tight',     # 移除多余的空白区域
                pad_inches=0,            # 设置边距为0
                transparent=True,         # 设置透明背景
                dpi=300)                 # 保持高分辨率
        plt.close()    
    

if __name__ == "__main__":
    visualizer = GreenVisualizer("testcases/json/13.json", "testcases/map/13.png")
    visualizer.plot()