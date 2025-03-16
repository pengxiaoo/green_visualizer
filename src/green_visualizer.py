import json
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import numpy as np

base_grid_num = 120
base_canvas_size = 10
smooth_sigma = 2
elevation_levels = 40
arrow_padding = 5
arrow_interval = 6
colors_gradient_list = [
    "#0000FF",  # blue
    "#00FF00",  # green
    "#FFFF00",  # yellow
    "#FFA500",  # orange
    "#FF0000",  # red
]


class GreenVisualizer:
    def __init__(self):
        self.data = None
        self.elevation_points = []
        self.green_border = None
        self.output_path = None

    def _load_json(self, json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def parse_data(self):
        """Parse JSON data, extract elevation points and boundary"""
        for feature in self.data["features"]:
            if feature["id"] == "Elevation":
                coords = feature["geometry"]["coordinates"]
                self.elevation_points.append(
                    {"x": coords[0], "y": coords[1], "z": coords[2]}
                )
            elif feature["id"] == "GreenBorder":
                coords = feature["geometry"]["coordinates"]
                self.green_border = Polygon(coords)

    def smooth_and_densify_edge(self) -> Polygon:
        """
        对边界进行加密处理
        基于平均距离进行线性插值加密
        Returns:
            Polygon: 加密后的边界多边形
        """
        boundary_polygon = self.green_border
        bx, by = boundary_polygon.exterior.xy
        boundary_points = np.column_stack([bx, by])
        
        # 计算相邻点之间的距离
        distances = np.sqrt(np.sum(np.diff(boundary_points, axis=0)**2, axis=1))
        d_avg = np.mean(distances)
        
        # 基于平均距离进行插值
        dense_points = []
        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]  # 循环到第一个点
            
            dense_points.append(p1)
            d = np.sqrt(np.sum((p2 - p1)**2))
            
            if d > d_avg:
                # 计算需要插入的点数
                n_points = int(d / d_avg) * 2
                # 生成插值点
                for j in range(1, n_points):
                    t = j / n_points
                    interpolated_point = p1 + t * (p2 - p1)
                    dense_points.append(interpolated_point)
        
        dense_points = np.array(dense_points)
        print(f"加密点数: {len(boundary_points)} -> {len(dense_points)}")
        return Polygon(dense_points)

    def plot_edge(self):
        xys = np.array([[p["x"], p["y"]] for p in self.elevation_points])
        zs = np.array([p["z"] for p in self.elevation_points])
        x_min, x_max = min(xys[:, 0]), max(xys[:, 0])
        y_min, y_max = min(xys[:, 1]), max(xys[:, 1])

        # 设置图形属性
        center_lat = (y_min + y_max) / 2
        center_lat_rad = np.pi * center_lat / 180
        aspect_ratio = 1 / np.cos(center_lat_rad)
        fig_width = base_canvas_size
        fig_height = int(base_canvas_size * aspect_ratio)
        _, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="none")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 绘制点
        plt.scatter(xys[:, 0], xys[:, 1], marker='+', label='Points')
        # 绘制边界
        smooth_polygon = self.smooth_and_densify_edge()
        bx, by = smooth_polygon.exterior.xy
        plt.scatter(bx, by, marker='o', label='Boundary', color='red')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            self.output_path,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=300,
        )
        plt.close()

    def plot(self):
        # Convert point data to numpy arrays
        xys = np.array([[p["x"], p["y"]] for p in self.elevation_points])
        zs = np.array([p["z"] for p in self.elevation_points])
        x_min, x_max = min(xys[:, 0]), max(xys[:, 0])
        y_min, y_max = min(xys[:, 1]), max(xys[:, 1])
        # Create 2d grid points with consistent spacing
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_grid_num = int(base_grid_num * (x_range / max(x_range, y_range)))
        y_grid_num = int(base_grid_num * (y_range / max(x_range, y_range)))
        xi = np.linspace(x_min, x_max, x_grid_num)
        yi = np.linspace(y_min, y_max, y_grid_num)
        xi, yi = np.meshgrid(xi, yi)

        # Adjust the aspect ratio based on the center latitude
        center_lat = (y_min + y_max) / 2
        center_lat_rad = np.pi * center_lat / 180
        aspect_ratio = 1 / np.cos(center_lat_rad)
        fig_width = base_canvas_size
        fig_height = int(base_canvas_size * aspect_ratio)
        _, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="none")
        ax.set_aspect("equal")

        # 使用平滑后的边界
        boundary_polygon = self.smooth_and_densify_edge()
        
        # Create mask
        mask = np.zeros_like(xi, dtype=bool)
        for i in range(xi.shape[0]):
            for j in range(xi.shape[1]):
                point = Point(xi[i, j], yi[i, j])
                mask[i, j] = boundary_polygon.contains(point) if boundary_polygon else True
        # Interpolation and mask application
        zi = griddata(xys, zs, (xi, yi), method="cubic")
        zi_masked = np.ma.masked_array(zi, ~mask)

        # Paint the color gradient
        levels = np.linspace(zs.min(), zs.max(), elevation_levels)
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            "custom", colors_gradient_list
        )
        ax.contourf(xi, yi, zi_masked, levels=levels, cmap=custom_cmap)

        # Paint contour lines
        ax.contour(xi, yi, zi_masked, levels=levels, colors="k", alpha=0.3)

        # Paint gradient arrows
        dx, dy = np.gradient(zi_masked)
        x_spacing = x_range / x_grid_num
        y_spacing = y_range / y_grid_num
        dx = dx / x_spacing
        dy = dy / y_spacing
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        dx_normalized = dx / magnitude
        dy_normalized = dy / magnitude
        skip = (
            slice(arrow_padding, -arrow_padding, arrow_interval),
            slice(arrow_padding, -arrow_padding, arrow_interval),
        )
        mask_skip = mask[skip]
        # ax.quiver: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html
        ax.quiver(
            xi[skip][mask_skip],  # x coordinate of start point
            yi[skip][mask_skip],  # y coordinate of start point
            -dy_normalized[skip][mask_skip],  # -dy/dx points to lower point
            -dx_normalized[skip][mask_skip],
            scale=15,  # global scale factor
            scale_units="width",
            units="width",
            width=0.005,  # base unit
            headwidth=8,  # arrow head width
            headlength=5,  # arrow head length
            headaxislength=2,  # arrow head bottom length
            minshaft=1,
            minlength=10,
            color="white",
            alpha=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            self.output_path,
            bbox_inches="tight",  # Remove extra white space
            pad_inches=0,  # Set margin to 0
            transparent=True,  # Set transparent background
            dpi=300,  # Keep high resolution
        )
        plt.close()

    def process_file(self, json_path, output_path):
        """Process single file"""
        self.data = self._load_json(json_path)
        self.elevation_points = []
        self.green_border = None
        self.output_path = output_path
        self.parse_data()
        self.plot_edge()


if __name__ == "__main__":
    visualizer = GreenVisualizer()
    try:
        for i in range(1, 19):
            json_file = f"testcases/json/{i}.json"
            png_file = json_file.replace(".json", "_edge.png").replace("/json", "/map")
            visualizer.process_file(json_file, png_file)
    finally:
        plt.close("all")
