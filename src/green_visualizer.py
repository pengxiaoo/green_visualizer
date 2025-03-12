import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as colors
import logging

smooth_sigma = 2

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
      
class GreenVisualizer:
    def __init__(self, json_path):
        self.data = self._load_json(json_path)
        self.elevation_points = []
        self.green_border = None
        self.parse_data()
        
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

    def create_visualization(self, resolution=100):
        """Create visualization of the green"""
        # Convert point data to numpy arrays
        points = np.array([[p['x'], p['y']] for p in self.elevation_points])
        values = np.array([p['z'] for p in self.elevation_points])
        
        # Create grid
        x_min, x_max = min(points[:,0]), max(points[:,0])
        y_min, y_max = min(points[:,1]), max(points[:,1])
        xi = np.linspace(x_min, x_max, resolution)
        yi = np.linspace(y_min, y_max, resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolation
        zi = griddata(points, values, (xi, yi), method='cubic')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Draw color gradient with custom colormap
        levels = np.linspace(values.min(), values.max(), 20)
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
        custom_cmap = colors.LinearSegmentedColormap.from_list('custom', colors_list)
        contour = ax.contourf(xi, yi, zi, levels=levels, cmap=custom_cmap)
        
        # 2. Draw contour lines
        ax.contour(xi, yi, zi, levels=levels, colors='k', alpha=0.3)
        
        # 3. Draw gradient arrows
        dx, dy = np.gradient(zi)
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(xi[skip], yi[skip], -dx[skip], -dy[skip], 
                 scale=50, color='white', alpha=0.5)
        
        # Add smoothed boundary
        if self.green_border:
            # Get boundary coordinates
            coords = list(self.green_border.exterior.coords)
            # Create smoothed polygon
            smooth_polygon = get_smooth_polygon(coords[:-1])  # Exclude last point as it's same as first
            if smooth_polygon:
                x, y = smooth_polygon.exterior.xy
                ax.plot(x, y, 'k-', linewidth=2)
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Elevation (m)')
        
        # Set title and axis labels
        ax.set_title('Green Topography')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        return fig, ax
    
    def save_plot(self, output_path='green_visualization.png'):
        """Save plot to file"""
        fig, _ = self.create_visualization()
        fig.savefig(output_path)
        plt.close(fig) 