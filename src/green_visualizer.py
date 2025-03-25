import json
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, splprep, splev

dpi = 300
target_meters_per_pixel = 0.02
lat_to_meter_ratio = 111000
base_grid_num = 400
# Added default arrow density parameter
arrow_spacing_in_meters = 2
# Increase density by sampling more points (higher the more sample points)
green_edge_sampling_factor = 3

colors_gradient_list = [
    "#1640C5",  # blue
    "#126ED4",  # light blue
    "#1C9AD9",  # medium blue
    "#0BBBCA",  # dark blue
    "#1AD7C6",  # sky blue
    "#3ADE8A",  # cyan
    "#4FE670",  # light aqua
    "#9AE639",  # medium sea green
    "#E1CF24",  # lime green
    "#E5A129",  # chartreuse
    "#E8862A",  # yellow
    "#E36626",  # light yellow
    "#F2451D",  # orange
    "#EF4123",  # dark orange
    "#EB3B2A",  # red orange
    "#CA253C",  # red
]
elevation_levels = len(colors_gradient_list)


class GreenVisualizer:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.data = None
        self.elevation_points = []
        self.green_border = None
        self.output_path = None
        self.xys = None
        self.zs = None
        self.xi = None
        self.yi = None
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.x_range = None
        self.y_range = None
        self.x_grid_num = None
        self.y_grid_num = None
        self.width_meters = None
        self.height_meters = None
        self.ax = None

    @staticmethod
    def _load_json(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def _init(self):
        for feature in self.data["features"]:
            if feature["id"] == "Elevation":
                coords = feature["geometry"]["coordinates"]
                self.elevation_points.append(
                    {"x": coords[0], "y": coords[1], "z": coords[2]}
                )
            elif feature["id"] == "GreenBorder":
                coords = feature["geometry"]["coordinates"]
                self.green_border = Polygon(coords)

        # Convert point data to numpy arrays for elevation points
        self.xys = np.array([[p["x"], p["y"]] for p in self.elevation_points])
        self.zs = np.array([p["z"] for p in self.elevation_points])

        # Smooth the green border
        self.green_border = self._smooth_and_densify_edge()

        # Now interpolate Z values for the border points
        border_points = np.array(self.green_border.exterior.coords)

        # First try linear interpolation for the borders
        interpolator = LinearNDInterpolator(self.xys, self.zs)
        border_z = interpolator(border_points[:, 0], border_points[:, 1])

        # Some points might be outside the convex hull of data points
        # Fill in any NaN values using nearest neighbor interpolation
        if np.any(np.isnan(border_z)):
            nearest_interp = NearestNDInterpolator(self.xys, self.zs)
            nan_indices = np.isnan(border_z)
            border_z[nan_indices] = nearest_interp(
                border_points[nan_indices, 0], border_points[nan_indices, 1]
            )

        # Combine original elevation points with the border points
        all_x = np.append(self.xys[:, 0], border_points[:, 0])
        all_y = np.append(self.xys[:, 1], border_points[:, 1])
        all_z = np.append(self.zs, border_z)

        # Update the point data arrays with the combined points
        self.xys = np.column_stack([all_x, all_y])
        self.zs = all_z

        adjustment_factor = 0.000001  # 0.11 meters adjustment
        self.x_min, self.x_max = min(self.xys[:, 0]) - adjustment_factor, max(self.xys[:, 0]) + adjustment_factor
        self.y_min, self.y_max = min(self.xys[:, 1]) - adjustment_factor, max(self.xys[:, 1]) + adjustment_factor

        # Create 2d grid points with consistent spacing
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.x_grid_num = int(
            base_grid_num * (self.x_range / max(self.x_range, self.y_range))
        )
        self.y_grid_num = int(
            base_grid_num * (self.y_range / max(self.x_range, self.y_range))
        )
        self.xi = np.linspace(self.x_min, self.x_max, self.x_grid_num)
        self.yi = np.linspace(self.y_min, self.y_max, self.y_grid_num)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

        # Set up the figure
        center_lat = (self.y_min + self.y_max) / 2
        center_lat_rad = np.pi * center_lat / 180
        self.width_meters = self.x_range * lat_to_meter_ratio * np.cos(center_lat_rad)
        self.height_meters = self.y_range * lat_to_meter_ratio

        # 计算需要的像素数
        pixels_width = int(self.width_meters / target_meters_per_pixel)
        pixels_height = int(self.height_meters / target_meters_per_pixel)

        # 计算所需的figure尺寸和dpi
        fig_width = pixels_width / dpi
        fig_height = pixels_height / dpi

        _, self.ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="none")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

    def _smooth_and_densify_edge(self) -> Polygon:
        """
        对边界进行加密处理
        基于平均距离进行线性插值加密
        Returns:
            Polygon: 加密后的边界多边形
        """

        boundary_polygon = self.green_border
        bx, by = boundary_polygon.exterior.xy

        # Remove consecutive duplicate points that might cause issues
        points = np.column_stack([bx, by])
        # Remove the last point if it's the same as the first (common in polygons)
        if np.array_equal(points[0], points[-1]):
            points = points[:-1]

        # Use splprep/splev which handles closed curves better
        # s=0 means no smoothing, just interpolation
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)

        # Generate new points along the spline
        # Increase density by sampling more points (u between 0 and 1)
        u_new = np.linspace(0, 1, len(points) * green_edge_sampling_factor, endpoint=False)

        # Evaluate the spline at the new points
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_points = np.column_stack([smooth_x, smooth_y])

        print(f"Smoothed points: {len(points)} -> {len(smooth_points)}")
        return Polygon(smooth_points)

    def _generate_masks(self):
        boundary_polygon = self.green_border
        # 利用boundary_polygon裁剪xi, yi
        mask = np.zeros_like(self.xi, dtype=bool)
        for i in range(self.xi.shape[0]):
            for j in range(self.xi.shape[1]):
                point = Point(self.xi[i, j], self.yi[i, j])
                mask[i, j] = boundary_polygon.contains(point)

        # 应用掩码但保持网格结构
        xi_masked = np.ma.masked_array(self.xi, ~mask)
        yi_masked = np.ma.masked_array(self.yi, ~mask)

        # 在边界附近增加插值点
        boundary_points = np.array(boundary_polygon.exterior.coords)
        # 获取边界上的高程值，使用nearest
        boundary_z = griddata(self.xys, self.zs, boundary_points, method="nearest")

        # 将边界点及其高程值添加到插值数据中
        xys_enhanced = np.vstack([self.xys, boundary_points])
        zs_enhanced = np.hstack([self.zs, boundary_z])

        # 首先使用linear方法进行插值，确保所有点都有值
        zi_linear = griddata(
            xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear"
        )

        # 然后使用cubic方法进行平滑
        valid_mask = ~np.isnan(zi_linear)
        if np.any(~valid_mask):
            print(f"发现 {np.sum(~valid_mask)} 个无效点，使用linear插值")
            zi = zi_linear
        else:
            # 只在有效区域内使用cubic插值
            zi = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic")
            # 如果cubic插值产生了nan值，回退到linear结果
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi[nan_mask] = zi_linear[nan_mask]

        zi_masked = np.ma.masked_array(zi, ~mask)
        return mask, xi_masked, yi_masked, zi_masked

    def _plot_edge(self):
        # 绘制点，并根据高程来着色
        plt.scatter(
            self.xys[:, 0],
            self.xys[:, 1],
            marker="+",
            label="Points",
            c=self.zs,
            cmap="viridis",
        )
        # 绘制边界
        smooth_polygon = self.green_border # already smoothed
        bx, by = smooth_polygon.exterior.xy
        plt.scatter(bx, by, marker="o", label="Boundary", color="red")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            f"{self.output_path.replace('.png', '_edge.png')}",
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=300,
        )
        plt.close()

    def _get_arrow_parameters(self):
        """
        Calculate arrow spacing and size based on density parameter
        to ensure even distribution with natural appearance.
        """
        # Calculate meters per grid cell
        meters_per_x_cell = self.width_meters / self.x_grid_num
        meters_per_y_cell = self.height_meters / self.y_grid_num

        # Calculate intervals based on physical dimensions and desired spacing
        x_arrow_interval = max(4, int(arrow_spacing_in_meters / meters_per_x_cell))
        y_arrow_interval = max(4, int(arrow_spacing_in_meters / meters_per_y_cell))

        # Calculate approximate number of arrows
        num_arrows_x = self.x_grid_num // x_arrow_interval
        num_arrows_y = self.y_grid_num // y_arrow_interval

        # Use square root for more natural scaling of arrow parameters with density
        density_factor = np.sqrt(1 / max(x_arrow_interval, y_arrow_interval))

        # Base width with more gentle scaling
        arrow_width = 0.01

        # Scale other arrow parameters with improved proportions
        arrow_headwidth = 6
        arrow_headlength = 6
        arrow_headaxislength = 6

        # Adjust arrow length scale for better appearance at lower densities
        arrow_length_scale_base = 70
        base_scale = arrow_length_scale_base * (1 + (1 - density_factor))
        arrow_length_scale = base_scale * density_factor

        return {
            'num_arrows_x': num_arrows_x,
            'num_arrows_y': num_arrows_y,
            'x_interval': x_arrow_interval,
            'y_interval': y_arrow_interval,
            'width': arrow_width,
            'headwidth': arrow_headwidth,
            'headlength': arrow_headlength,
            'headaxislength': arrow_headaxislength,
            'length_scale': arrow_length_scale
        }

    def _eps_gradient(self, zi):
        epsilon = 1e-8
        gradient_y, gradient_x = np.gradient(zi)
        magnitude = np.hypot(gradient_x, gradient_y)
        magnitude = np.where(magnitude < epsilon, epsilon, magnitude)
        return -gradient_x / magnitude, -gradient_y / magnitude

    def _plot(self):
        # 生成掩码和插值结果
        mask, xi_masked, yi_masked, zi_masked = self._generate_masks()

        # Paint the color gradient
        levels = np.linspace(self.zs.min(), self.zs.max(), elevation_levels)
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            "custom", colors_gradient_list
        )
        self.ax.contourf(
            xi_masked, yi_masked, zi_masked, levels=levels, cmap=custom_cmap
        )

        # Plot the green border
        bx, by = self.green_border.exterior.xy
        self.ax.plot(bx, by, color="black", linewidth=1.3)

        arrows_params = self._get_arrow_parameters()

        # Calculate gradient for arrows & arrow grid creation
        dx, dy = self._eps_gradient(zi_masked)
        y_idx = np.linspace(0, self.xi.shape[0] - 1, arrows_params["num_arrows_x"], dtype=int)
        x_idx = np.linspace(0, self.xi.shape[1] - 1, arrows_params["num_arrows_y"], dtype=int)

        indices = np.ix_(y_idx, x_idx)

        X = self.xi[indices]
        Y = self.yi[indices]
        U = dx[indices]
        V = dy[indices]

        buffered = self.green_border.buffer(-1e-5)
        valid = np.array([
            buffered.contains(Point(x, y))
            for x, y in zip(X.ravel(), Y.ravel())
        ]).reshape(X.shape)

        self.ax.quiver(
            X[valid],
            Y[valid],
            U[valid],
            V[valid],
            color="black",
            scale=arrows_params["length_scale"],
            width=arrows_params["width"],
            headwidth=arrows_params["headwidth"],
            headlength=arrows_params["headlength"],
            headaxislength=arrows_params["headaxislength"],
            minshaft=1.8,
            pivot="middle",
        )
        plt.savefig(
            self.output_path,
            bbox_inches="tight",  # Remove extra white space
            pad_inches=0,  # Set margin to 0
            transparent=True,  # Set transparent background
            dpi=dpi,  # Keep high resolution
        )
        plt.close()

    def process_file(self, json_path, output_path):
        """Process single file"""
        self._reset()
        self.output_path = output_path
        self.data = self._load_json(json_path)
        self._init()
        self._plot()
        self._plot_edge()
        plt.close()


if __name__ == "__main__":
    visualizer = GreenVisualizer()
    try:
        # run all the 18 testcases
        for i in range(1, 19):
            json_file = f"testcases/json/{i}.json"
            png_file = json_file.replace(".json", ".png").replace("/json", "/map")
            visualizer.process_file(json_file, png_file)
    finally:
        plt.close("all")
