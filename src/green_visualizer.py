import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as colors

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
    
    def _smooth_boundary(self, x, y, window_size=5):
        """
        Smooth boundary coordinates using moving average.
        
        Args:
            x (array-like): X coordinates of the boundary
            y (array-like): Y coordinates of the boundary
            window_size (int): Size of the smoothing window
            
        Returns:
            tuple: Smoothed x and y coordinates
        """
        x, y = np.array(x), np.array(y)
        x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
        y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        return x_smooth, y_smooth

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
            '#FFFF00',  # yellow
            '#FFA500',  # orange
            '#FF0000'   # red
        ]
        custom_cmap = colors.LinearSegmentedColormap.from_list('custom', colors_list)
        contour = ax.contourf(xi, yi, zi, levels=levels, cmap=custom_cmap)
        
        # 2. Draw contour lines
        ax.contour(xi, yi, zi, levels=levels, colors='k', alpha=0.3)
        
        # 3. Draw gradient arrows
        dx, dy = np.gradient(zi)
        # Downsample to reduce arrow density
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(xi[skip], yi[skip], -dx[skip], -dy[skip], 
                 scale=50, color='white', alpha=0.5)
        
        # Add smoothed boundary
        if self.green_border:
            # Get and smooth boundary coordinates
            x, y = self.green_border.exterior.xy
            x_smooth, y_smooth = self._smooth_boundary(x, y)
            # Plot smoothed boundary
            ax.plot(x_smooth, y_smooth, 'k-', linewidth=2)
        
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