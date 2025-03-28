import json

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, splprep, splev
from pyproj import Transformer, CRS

input_crs = CRS.from_string('EPSG:4326')
output_crs = CRS.from_string('EPSG:3857')
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

dpi = 300
target_meters_per_pixel = 0.02
base_grid_num = 400
# Added default arrow density parameter
arrow_spacing_in_meters = 6
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
        self.transformer = None
        self.arrow_spacing_in_meters = None
        self.adj_ratio = None

    def _transform_coordinates(self, coords):
        if isinstance(coords, list):
            coords = np.array(coords)

        if coords.ndim == 1:
            x, y = transformer.transform(coords[0], coords[1])
            return np.array([x, y])
        else:
            transformed = np.array([transformer.transform(lon, lat) for lon, lat in coords])
            return transformed

    @staticmethod
    def _load_json(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def _init(self):
        for feature in self.data["features"]:
            if feature["id"] == "Elevation":
                coords = feature["geometry"]["coordinates"]
                transformed_xy = self._transform_coordinates(coords[:2])
                self.elevation_points.append(
                    {"x": transformed_xy[0], "y": transformed_xy[1], "z": coords[2]}
                )
            elif feature["id"] == "GreenBorder":
                coords = feature["geometry"]["coordinates"]
                transformed_coords = self._transform_coordinates(coords)
                self.green_border = Polygon(transformed_coords)

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

        adjustment_factor = 0.11 # 0.11 meters adjustment
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
        self.width_meters = self.x_range
        self.height_meters = self.y_range

        # 计算需要的像素数
        pixels_width = int(self.width_meters / target_meters_per_pixel)
        pixels_height = int(self.height_meters / target_meters_per_pixel)

        # 计算所需的figure尺寸和dpi
        fig_width = pixels_width / dpi
        fig_height = pixels_height / dpi

        self.adj_ratio = self.width_meters / self.height_meters

        if self.adj_ratio < 0.5:
            self.arrow_spacing_in_meters = 3
        else:
            self.arrow_spacing_in_meters = arrow_spacing_in_meters

        print(f"{self.width_meters}, {self.height_meters}, {pixels_width}, {pixels_height}, {fig_width}, {fig_height}")

        _, self.ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="none")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.set_aspect("equal", adjustable='box')
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
        boundary_z = griddata(self.xys, self.zs, boundary_points, method="nearest")

        # 将边界点及其高程值添加到插值数据中
        xys_enhanced = np.vstack([self.xys, boundary_points])
        zs_enhanced = np.hstack([self.zs, boundary_z])

        # Create Z value for Contour line drawing
        # 首先使用linear方法进行插值，确保所有点都有值
        zi_linear_contour = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear")
        # 然后使用cubic方法进行平滑
        valid_mask = ~np.isnan(zi_linear_contour)
        if np.any(~valid_mask):
            print(f"发现 {np.sum(~valid_mask)} 个无效点，使用linear插值")
            zi_contour = zi_linear_contour
        else:
            # 只在有效区域内使用cubic插值
            zi_contour = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic")
            # 如果cubic插值产生了nan值，回退到linear结果
            nan_mask = np.isnan(zi_contour)
            if np.any(nan_mask):
                zi_contour[nan_mask] = zi_linear_contour[nan_mask]
        zi_masked_contour = np.ma.masked_array(zi_contour, ~mask)

        # Create Z values for Arrow Extrapolation
        # Interpolate using 'linear' first
        zi_linear = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear")

        nan_mask = np.isnan(zi_linear)

        if np.any(nan_mask):
            print(f"发现 {np.sum(nan_mask)} 个无效点，执行外推")

            zi_nearest = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="nearest")
            zi_linear[nan_mask] = zi_nearest[nan_mask]

        zi_cubic = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic")

        zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)

        zi_masked = np.ma.masked_array(zi, ~mask)
        return mask, xi_masked, yi_masked, zi_masked, zi_masked_contour

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
        Calculate arrow spacing and size based on input data characteristics
        with adaptive scaling for different rectangle/square sizes.

        Aims to maintain consistent visual representation across different data ranges.
        """
        # Total area of the region in square meters
        total_area = self.width_meters * self.height_meters

        # Desired physical spacing between arrows in meters
        desired_arrow_spacing = self.arrow_spacing_in_meters

        # Calculate number of arrows based on area and desired spacing
        # Use square root to prevent exponential growth
        estimated_arrow_count = int(np.sqrt(total_area / (desired_arrow_spacing ** 2)))

        # Adjust parameters based on data range and total area
        # Normalization factors to provide consistent behavior
        area_normalization_factor = np.log1p(total_area) / 10  # Logarithmic scaling

        # Arrow width - proportional to the smallest cell dimension
        min_cell_size = min(self.width_meters / self.x_grid_num,
                            self.height_meters / self.y_grid_num)
        arrow_width = max(0.01, min_cell_size / 100)  # Ensure minimum visibility

        # Arrow length scaling
        # Adjust based on total area and desired spacing
        if self.adj_ratio >= 1.5:
            arrow_head_param = 6
            base_arrow_length_scale = 60
        elif self.adj_ratio >= 0.9:
            arrow_head_param = 5
            base_arrow_length_scale = 50
        elif self.adj_ratio >= 0.8:
            arrow_head_param = 4
            base_arrow_length_scale = 50
        else:
            arrow_head_param = 6.5
            base_arrow_length_scale = 35
        arrow_length_scale = base_arrow_length_scale * area_normalization_factor

        # Head parameters - proportional to arrow width
        arrow_headwidth = arrow_head_param
        arrow_headlength = arrow_head_param
        arrow_headaxislength = arrow_head_param

        # Density factor - how crowded the arrows should be
        density_factor = np.clip(
            np.sqrt(estimated_arrow_count) / 10,  # Soft normalization
            0.2,  # Minimum density
            2.0  # Maximum density
        )

        length_scale = arrow_length_scale * density_factor

        # Final parameter adjustments
        return {
            'x_interval': max(2, int(desired_arrow_spacing / min_cell_size)),
            'y_interval': max(2, int(desired_arrow_spacing / min_cell_size)),
            'width': arrow_width,
            'headwidth': arrow_headwidth,
            'headlength': arrow_headlength,
            'headaxislength': arrow_headaxislength,
            'length_scale': length_scale,
            'estimated_arrow_count': estimated_arrow_count
        }

    def _eps_gradient(self, zi):
        epsilon = 1e-8
        gradient_y, gradient_x = np.gradient(zi)
        magnitude = np.hypot(gradient_x, gradient_y)
        magnitude = np.where(magnitude < epsilon, epsilon, magnitude)
        return -gradient_x / magnitude, -gradient_y / magnitude

    def _rectangular_fit_score(self, polygon, threshold=0.8):
        minx, miny, maxx, maxy = polygon.bounds
        bounding_box_area = (maxx - minx) * (maxy - miny)

        if bounding_box_area == 0:
            return 0, False  # Prevent division by zero

        score = polygon.area / bounding_box_area
        is_rectangular = score >= threshold
        return score, is_rectangular

    def _plot(self):
        # 生成掩码和插值结果
        mask, xi_masked, yi_masked, zi_masked, zi_masked_contour = self._generate_masks()

        # Paint the color gradient
        levels = np.linspace(self.zs.min(), self.zs.max(), elevation_levels)
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            "custom", colors_gradient_list
        )
        self.ax.contourf(
            xi_masked, yi_masked, zi_masked_contour, levels=levels, cmap=custom_cmap
        )

        # Plot the green border
        bx, by = self.green_border.exterior.xy
        self.ax.plot(bx, by, color="black", linewidth=3)
        polygon_path = Path(np.column_stack((bx, by)))
        clip_patch = PathPatch(polygon_path,
                               transform=self.ax.transData,
                               facecolor='none',
                               edgecolor='none')

        arrows_params = self._get_arrow_parameters()
        print(f"Arrow parameters: {arrows_params}")

        # Calculate gradient for arrows & arrow grid creation
        dx, dy = self._eps_gradient(zi_masked)


        x_grid = np.arange(self.x_min, self.x_max, self.arrow_spacing_in_meters)
        y_grid = np.arange(self.y_min, self.y_max, self.arrow_spacing_in_meters)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Flatten original grid coordinates
        xi_flat = self.xi.flatten()
        yi_flat = self.yi.flatten()

        # Interpolate U and V vectors
        U = griddata((xi_flat, yi_flat), dx.flatten(), (X, Y), method='linear', fill_value=0)
        V = griddata((xi_flat, yi_flat), dy.flatten(), (X, Y), method='linear', fill_value=0)

        # Normalize U, V for uniform arrow size
        magnitude = np.sqrt(U ** 2 + V ** 2)
        eps = 1e-8
        U = (U / (magnitude + eps))
        V = (V / (magnitude + eps))

        # Filter points within green border
        score, is_rectangular = self._rectangular_fit_score(self.green_border)
        if is_rectangular:
            buffer_length = -0.6
        else:
            buffer_length = -1.665
        buffered = self.green_border.buffer(buffer_length)
        valid = np.array([
            buffered.covers(Point(x, y))
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
            clip_path=clip_patch,
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
        plt.close()


if __name__ == "__main__":
    visualizer = GreenVisualizer()
    try:
        # run all the testcases
        for course_index in range(1, 4):
            for hole_index in range(1, 19):
                json_file = f"testcases/json/course{course_index}/{hole_index}.json"
                png_file = json_file.replace(".json", ".png").replace("/json", "/map")
                visualizer.process_file(json_file, png_file)
    finally:
        plt.close("all")
