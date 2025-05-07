from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Point, Polygon, MultiPoint
from scipy.interpolate import (
    griddata,
    LinearNDInterpolator,
    NearestNDInterpolator,
    splprep,
    splev,
)
import os
import numpy as np
import json
from utils import (
    logger,
    dpi,
    transform_coordinates,
)
import utils

debug = False
base_grid_num = 400
pixel_scale = 1.3
meters_per_pixel = utils.meters_per_pixel / 2
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
# Added default arrow density parameter
arrow_spacing_in_meters = 5
arrow_head_width = 5


def get_arrow_width_and_length_scale(xy_ratio, sqrt_area):
    arrow_params = OrderedDict(
        [  # ratio_threshold, (arrow_width, arrow_length_scale)
            (0.40, (0.0145, 6.5)),
            (0.50, (0.0120, 7.5)),
            (0.60, (0.0110, 8.0)),
            (0.70, (0.0103, 8.5)),
            (0.80, (0.0100, 8.8)),
            (0.95, (0.0090, 9.8)),
            (1.05, (0.0087, 10.3)),
            (1.10, (0.0085, 10.5)),
            (1.20, (0.0082, 10.8)),
            (1.30, (0.0080, 11.1)),
            (1.40, (0.0080, 11.5)),
            (1.50, (0.0078, 11.8)),
            (1.60, (0.0072, 12.1)),
            (1.70, (0.0071, 12.4)),
            (1.80, (0.0070, 12.8)),
            (1.90, (0.0069, 13.0)),
            (2.00, (0.0068, 13.3)),
        ]
    )
    arrow_width, arrow_length_scale = 0.0066, 13.6
    for ratio_threshold, params in arrow_params.items():
        if xy_ratio <= ratio_threshold:
            arrow_width, arrow_length_scale = params
            break
    adjuster = max(1, sqrt_area / 45)
    return arrow_width / adjuster, arrow_length_scale * adjuster


class GreenVisualizer:
    def __init__(self):
        self.ax = None
        self.fig = None
        self.data = None
        self.green_border = None
        self.xys = None
        self.zs = None
        self.xi = None
        self.yi = None

    def cleanup(self):
        """清理资源"""
        if self.ax is not None:
            self.ax.clear()
            self.ax = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        self.data = None
        self.green_border = None
        self.xys = None
        self.zs = None
        self.xi = None
        self.yi = None
        plt.close("all")  # 确保所有图形都被关闭

    @staticmethod
    def _load_json(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def _init(
        self,
        hole: dict,
        club_id: str,
        course_id: str,
        output_path: str,
    ):
        self.output_path = output_path
        self.club_id = club_id
        self.course_id = course_id
        self.hole_number = hole["hole_number"]
        self.elevation_points = []
        for feature in hole["features"]:
            if feature["id"] == "Elevation":
                coords = feature["geometry"]["coordinates"]
                transformed_xy = transform_coordinates(coords[:2])
                self.elevation_points.append(
                    {"x": transformed_xy[0], "y": transformed_xy[1], "z": coords[2]}
                )
            elif feature["id"] == "GreenBorder":
                coords = feature["geometry"]["coordinates"]
                self.green_boundaries_raw = [coords]
                self.bounds_latlon = utils.get_smooth_polygon(coords).bounds
                transformed_coords = transform_coordinates(coords)
                self.green_border = Polygon(transformed_coords)

        # Convert point data to numpy arrays for elevation points
        self.xys = np.array([[p["x"], p["y"]] for p in self.elevation_points])
        self.zs = np.array([p["z"] for p in self.elevation_points])
        self.polygon = MultiPoint(self.xys).convex_hull

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

        self.x_min, self.x_max = min(self.xys[:, 0]), max(self.xys[:, 0])
        self.y_min, self.y_max = min(self.xys[:, 1]), max(self.xys[:, 1])

        # Create 2d grid points with consistent spacing
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        x_grid_num = int(
            base_grid_num * (self.x_range / max(self.x_range, self.y_range))
        )
        y_grid_num = int(
            base_grid_num * (self.y_range / max(self.x_range, self.y_range))
        )
        self.xi = np.linspace(self.x_min, self.x_max, x_grid_num)
        self.yi = np.linspace(self.y_min, self.y_max, y_grid_num)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

        # Set up the figure
        self.width_meters = self.x_range
        self.height_meters = self.y_range
        self.total_area = self.width_meters * self.height_meters
        self.sqrt_area = np.sqrt(self.total_area)

        # 计算需要的像素数
        self.pixels_width = int(self.width_meters / meters_per_pixel)
        self.pixels_height = int(self.height_meters / meters_per_pixel)

        # 计算所需的figure尺寸和dpi
        fig_width = self.pixels_width / dpi
        fig_height = self.pixels_height / dpi

        self.xy_ratio = self.width_meters / self.height_meters

        logger.info(
            f"{self.width_meters}, {self.height_meters}, {self.pixels_width}, {self.pixels_height}, {fig_width}, {fig_height}"
        )

        _, self.ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="none")
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.set_aspect("equal", adjustable="box")
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
        u_new = np.linspace(
            0, 1, len(points) * green_edge_sampling_factor, endpoint=False
        )

        # Evaluate the spline at the new points
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_points = np.column_stack([smooth_x, smooth_y])

        logger.info(f"Smoothed points: {len(points)} -> {len(smooth_points)}")
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
        zi_linear_contour = griddata(
            xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear"
        )
        # 然后使用cubic方法进行平滑
        valid_mask = ~np.isnan(zi_linear_contour)
        if np.any(~valid_mask):
            logger.warning(f"发现 {np.sum(~valid_mask)} 个无效点，使用linear插值")
            zi_contour = zi_linear_contour
        else:
            # 只在有效区域内使用cubic插值
            zi_contour = griddata(
                xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic"
            )
            # 如果cubic插值产生了nan值，回退到linear结果
            nan_mask = np.isnan(zi_contour)
            if np.any(nan_mask):
                zi_contour[nan_mask] = zi_linear_contour[nan_mask]
        zi_masked_contour = np.ma.masked_array(zi_contour, ~mask)

        # Create Z values for Arrow Extrapolation
        # Interpolate using "linear" first
        zi_linear = griddata(
            xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear"
        )

        nan_mask = np.isnan(zi_linear)

        if np.any(nan_mask):
            logger.info(f"发现 {np.sum(nan_mask)} 个无效点，执行外推")

            zi_nearest = griddata(
                xys_enhanced, zs_enhanced, (self.xi, self.yi), method="nearest"
            )
            zi_linear[nan_mask] = zi_nearest[nan_mask]

        zi_cubic = griddata(
            xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic"
        )

        zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)

        zi_masked = np.ma.masked_array(zi, ~mask)
        return xi_masked, yi_masked, zi_masked, zi_masked_contour

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
        smooth_polygon = self.green_border  # already smoothed
        bx, by = smooth_polygon.exterior.xy
        # intersaction between green_border and xys
        intersection_polygon = smooth_polygon.intersection(self.polygon)
        if intersection_polygon.is_empty:
            logger.error(f"hole {self.hole_number} green boundary 与 xys 没有相交区域")
        else:
            logger.info(f"hole {self.hole_number} green boundary 与 xys 相交的区域面积为 {intersection_polygon.area}")

        plt.scatter(bx, by, marker="o", label="Boundary", color="red")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        output_png_path = f"{self.output_path}/{self.hole_number}_edge.png"
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        plt.savefig(
            output_png_path,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=dpi,
        )
        plt.close()

    def _eps_gradient(self, zi):
        epsilon = 1e-8
        gradient_y, gradient_x = np.gradient(zi)
        magnitude = np.hypot(gradient_x, gradient_y)
        magnitude = np.where(magnitude < epsilon, epsilon, magnitude)
        return -gradient_x / magnitude, -gradient_y / magnitude

    def _plot(self):
        intersection_polygon = self.green_border.intersection(self.polygon)
        if intersection_polygon.is_empty:
            raise Exception(
                f"hole {self.hole_number} green boundary 与 xys 没有相交区域"
            )
        # 生成掩码和插值结果
        xi_masked, yi_masked, zi_masked, zi_masked_contour = self._generate_masks()

        # Paint the color gradient
        levels = np.linspace(self.zs.min(), self.zs.max(), len(colors_gradient_list))
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            "custom", colors_gradient_list
        )
        self.ax.contourf(
            xi_masked, yi_masked, zi_masked_contour, levels=levels, cmap=custom_cmap
        )

        # Plot the green border
        shrunked_border = self.green_border.buffer(-0.1)
        bx, by = shrunked_border.exterior.xy
        self.ax.plot(bx, by, color="black", linewidth=1)
        polygon_path = Path(np.column_stack((bx, by)))
        clip_patch = PathPatch(
            polygon_path,
            transform=self.ax.transData,
            facecolor="none",
            edgecolor="none",
        )

        # Calculate gradient for arrows & arrow grid creation
        dx, dy = self._eps_gradient(zi_masked)

        x_grid = np.arange(self.x_min, self.x_max, arrow_spacing_in_meters)
        y_grid = np.arange(self.y_min, self.y_max, arrow_spacing_in_meters)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Flatten original grid coordinates
        xi_flat = self.xi.flatten()
        yi_flat = self.yi.flatten()

        # Interpolate U and V vectors
        U = griddata(
            (xi_flat, yi_flat), dx.flatten(), (X, Y), method="linear", fill_value=0
        )
        V = griddata(
            (xi_flat, yi_flat), dy.flatten(), (X, Y), method="linear", fill_value=0
        )

        # Normalize U, V for uniform arrow size
        magnitude = np.sqrt(U**2 + V**2)
        eps = 1e-8
        U = U / (magnitude + eps)
        V = V / (magnitude + eps)

        valid = np.array(
            [
                self.green_border.covers(Point(x, y))
                for x, y in zip(X.ravel(), Y.ravel())
            ]
        ).reshape(X.shape)

        arrow_width, length_scale = get_arrow_width_and_length_scale(
            self.xy_ratio, self.sqrt_area
        )
        self.ax.quiver(
            X[valid],
            Y[valid],
            U[valid],
            V[valid],
            color="black",
            scale=length_scale,
            width=arrow_width,
            headwidth=arrow_head_width,
            headlength=arrow_head_width,
            headaxislength=arrow_head_width,
            minshaft=1.8,
            pivot="middle",
            clip_path=clip_patch,
        )

        if debug:
            # ===在左下角添加文字===
            self.ax.text(
                self.x_min + 0.1,
                self.y_min + 0.1,
                f"xy_ratio: {self.xy_ratio:.2f}\nsqrt_area: {self.sqrt_area:.2f}\narrow_width: {arrow_width:.4f}\nlength_scale: {length_scale:.3f}",
                fontsize=10,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5),
            )
        output_png_path = f"{self.output_path}/{self.hole_number}.png"
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        plt.savefig(
            output_png_path,
            bbox_inches="tight",  # Remove extra white space
            pad_inches=0,  # Set margin to 0
            transparent=True,  # Set transparent background
            dpi=dpi,  # Keep high resolution
        )
        plt.close()

    def plot_holes(
        self, club_id: str, course_id: str, holes: list[dict], output_path: str
    ) -> str:
        # 先创建目录
        os.makedirs(output_path, exist_ok=True)
        try:
            permissions = oct(os.stat(output_path).st_mode)[-3:]
            logger.warning(f"Output directory permissions: {permissions}")
        except Exception as e:
            logger.warning(f"Failed to check directory permissions: {e}")

        # 继续后续的绘图操作
        logger.warning(f"Current working directory: {os.getcwd()}")
        logger.warning(f"Output directory exists: {os.path.exists(output_path)}")
        for hole in holes:
            try:
                self._init(hole, club_id, course_id, output_path)
                self._plot()
                # self._plot_edge()
            except Exception as e:
                logger.error(f"hole {hole['hole_number']} 绘制失败: {e}")
            finally:
                self.cleanup()  # 确保资源被清理


if __name__ == "__main__":
    visualizer = GreenVisualizer()
    try:
        # run all the testcases
        holes = []
        for course_index in range(1, 4):
            for hole_index in range(1, 19):
                json_file = f"testcases/input/course{course_index}/{hole_index}.json"
                hole = json.load(open(json_file, "r", encoding="utf-8"))
                hole["hole_number"] = hole_index
                holes.append(hole)
            visualizer.plot_holes(
                "", "", holes, f"testcases/output/green_2d/course{course_index}"
            )
    finally:
        visualizer.cleanup()  # 确保最后清理资源
