import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as colors
from alphashape import alphashape

resolution = 100
base_size = 10
smooth_sigma = 2
alpha = 0.1 # alpha值越大，形状越接近凸包; alpha值越小，形状越"贴合"点集
elevation_levels = 40
colors_gradient_list = [
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

    def plot(self):
        """Create visualization of the green"""
        # Convert point data to numpy arrays
        xys = np.array([[p['x'], p['y']] for p in self.elevation_points])
        zs = np.array([p['z'] for p in self.elevation_points])
        
        # Create 2d grid points with consistent spacing
        x_min, x_max = min(xys[:,0]), max(xys[:,0])
        y_min, y_max = min(xys[:,1]), max(xys[:,1])
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_resolution = int(resolution * (x_range / max(x_range, y_range)))
        y_resolution = int(resolution * (y_range / max(x_range, y_range)))
        xi = np.linspace(x_min, x_max, x_resolution)
        yi = np.linspace(y_min, y_max, y_resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # Adjust the aspect ratio based on the center latitude
        center_lat = y_min + y_max / 2
        center_lat_rad = np.pi * center_lat / 180
        aspect_ratio = 1 / np.cos(center_lat_rad)
        fig_width = base_size
        fig_height = base_size * aspect_ratio
        _, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='none')
        ax.set_aspect('equal') 
        
        # todo: Improve the edge smoothing
        points = [(x, y) for x, y in xys]
        alpha_shape = alphashape(points, alpha)
        # If alpha shape is MultiPolygon, take the largest one
        if alpha_shape.geom_type == 'MultiPolygon':
            alpha_shape = max(alpha_shape, key=lambda x: x.area)
        boundary_coords = list(alpha_shape.exterior.coords)
        smoothed_edge_points = smooth_coordinates(boundary_coords)
        smooth_border = get_smooth_polygon(smoothed_edge_points)
        # Create mask
        mask = np.zeros_like(xi, dtype=bool)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                point = Point(xi[i,j], yi[i,j])
                mask[i,j] = smooth_border.contains(point) if smooth_border else True
        # Interpolation and mask application
        zi = griddata(xys, zs, (xi, yi), method='cubic')
        zi_masked = np.ma.masked_array(zi, ~mask)
        
        # Paint the color gradient
        levels = np.linspace(zs.min(), zs.max(), elevation_levels)
        custom_cmap = colors.LinearSegmentedColormap.from_list('custom', colors_gradient_list)
        contour = ax.contourf(xi, yi, zi_masked, levels=levels, cmap=custom_cmap)
        
        # Paint contour lines
        ax.contour(xi, yi, zi_masked, levels=levels, colors='k', alpha=0.3)
        
        # todo: gradient arrows looks wrong
        # Calculate and print gradient information
        dx, dy = np.gradient(zi_masked)
        # Normalize all arrows to unit vectors
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitude = np.where(magnitude == 0, 1, magnitude)
        dx_normalized = dx / magnitude
        dy_normalized = dy / magnitude
        skip = (slice(None, None, 6), slice(None, None, 6))
        mask_skip = mask[skip]
        # ax.quiver: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html
        ax.quiver(xi[skip][mask_skip], yi[skip][mask_skip], 
                 -dx_normalized[skip][mask_skip], -dy_normalized[skip][mask_skip], 
                 scale=15,          # Global scale factor
                 scale_units='width',
                 units='width',
                 width=0.005,       # Base unit
                 headwidth=8,       # Arrow head width
                 headlength=5,      # Arrow head length
                 headaxislength=2,  # Arrow head bottom length
                 minshaft=1,        
                 minlength=10,     
                 color='white', 
                 alpha=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(self.output_path, 
                bbox_inches='tight',     # Remove extra white space
                pad_inches=0,            # Set margin to 0
                transparent=True,         # Set transparent background
                dpi=300)                 # Keep high resolution
        plt.close()    
    

if __name__ == "__main__":
    json_file = "testcases/json/13.json"
    png_file = json_file.replace(".json", ".png").replace("/json", "/map")
    visualizer = GreenVisualizer(json_file, png_file)
    visualizer.plot()