import json
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator, splprep, splev
from pyproj import Transformer, CRS
import numpy as np
from scipy.ndimage import gaussian_filter  # 引入高斯滤波器来实现卷积插值
import os

# -------------------------- 系统级配置参数 --------------------------
input_crs = CRS.from_string('EPSG:4326')
output_crs = CRS.from_string('EPSG:3857')
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

dpi = 300
target_meters_per_pixel = 0.02  # 控制像素分辨率，每个像素对应实际的地面距离
base_grid_num = 400
arrow_spacing_in_meters = 6  # 默认箭头密度参数
green_edge_sampling_factor = 3  # 边界采样因子
grid_density = 16  # 每平方米的网格数量（修改为16个网格点）

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
    """增强版地理可视化处理器（针对箭头渲染、边界采样及数据转换进行优化）"""

    def __init__(self):
        self._reset()

    def _reset(self):
        """重置所有实例状态"""
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
        self.dx = None  # 网格X方向间距（米）
        self.dy = None  # 网格Y方向间距（米）
        self.ax = None
        self.transformer = None
        self.pixel_count = None  # 图片像素总数
        self.pixels_width = None  # 图片像素宽度
        self.pixels_height = None  # 图片像素高度
        self.a = None  # 像素/网格比
        self.L = None  # 修正系数
        self.arrow_spacing_in_meters = None  # 箭头间距（米）
        self.adj_ratio = None  # 宽高比
        self.arrow_count = 0  # 箭头数量统计

    def _transform_coordinates(self, coords):
        """转换地理坐标系，将EPSG:4326转换为EPSG:3857"""
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
        """加载GeoJSON文件"""
        with open(json_path, "r") as f:
            return json.load(f)

    def _init(self):
        """初始化地理数据和网格参数"""
        # 遍历GeoJSON中的每个要素，分别处理高程数据和绿色边界数据
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

        # 构造高程数据的数组
        self.xys = np.array([[p["x"], p["y"]] for p in self.elevation_points])
        self.zs = np.array([p["z"] for p in self.elevation_points])

        # 对绿色边界进行平滑和密化
        self.green_border = self._smooth_and_densify_edge()
        border_points = np.array(self.green_border.exterior.coords)

        # 使用LinearNDInterpolator对边界点的高程进行初步估算
        interpolator = LinearNDInterpolator(self.xys, self.zs)
        border_z = interpolator(border_points[:, 0], border_points[:, 1])

        # 如果存在无效值，则采用NearestNDInterpolator进行补充
        if np.any(np.isnan(border_z)):
            nearest_interp = NearestNDInterpolator(self.xys, self.zs)
            nan_indices = np.isnan(border_z)
            border_z[nan_indices] = nearest_interp(border_points[nan_indices, 0], border_points[nan_indices, 1])

        # 将原始高程点与边界采样点合并
        all_x = np.append(self.xys[:, 0], border_points[:, 0])
        all_y = np.append(self.xys[:, 1], border_points[:, 1])
        all_z = np.append(self.zs, border_z)

        self.xys = np.column_stack([all_x, all_y])
        self.zs = all_z

        # 设置边界缓冲，确保数据范围略大于实际数据
        adjustment_factor = 0.11
        self.x_min, self.x_max = min(self.xys[:, 0]) - adjustment_factor, max(self.xys[:, 0]) + adjustment_factor
        self.y_min, self.y_max = min(self.xys[:, 1]) - adjustment_factor, max(self.xys[:, 1]) + adjustment_factor

        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

        # 计算图像的物理尺寸（单位：米）
        self.width_meters = self.x_range
        self.height_meters = self.y_range

        # 计算图像面积，并根据每平方米的网格数量确定网格总数
        image_area = self.width_meters * self.height_meters
        total_grids = int(image_area * grid_density)
        grid_ratio = np.sqrt(total_grids / (self.x_range * self.y_range))

        # 重新计算网格数目
        self.x_grid_num = int(self.x_range * grid_ratio)
        self.y_grid_num = int(self.y_range * grid_ratio)

        # 构建规则网格
        self.xi = np.linspace(self.x_min, self.x_max, self.x_grid_num)
        self.yi = np.linspace(self.y_min, self.y_max, self.y_grid_num)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

        # 计算网格间距（米）
        self.dx = self.x_range / (self.x_grid_num - 1) if self.x_grid_num > 1 else 0
        self.dy = self.y_range / (self.y_grid_num - 1) if self.y_grid_num > 1 else 0

        # 根据目标分辨率计算图像像素尺寸
        self.pixels_width = int(self.width_meters / target_meters_per_pixel)
        self.pixels_height = int(self.height_meters / target_meters_per_pixel)
        self.pixel_count = self.pixels_width * self.pixels_height

        # 计算像素与网格数的比例，用于后续箭头参数的调整
        self.a = self.pixel_count / (self.x_grid_num * self.y_grid_num)
        self.L = (self.a - 156)  # 修正系数（百分比调整）

        # 设置箭头间距参数，依据图幅宽高比自适应调整
        self.adj_ratio = self.width_meters / self.height_meters
        if self.adj_ratio < 0.5:
            self.arrow_spacing_in_meters = 3
        else:
            self.arrow_spacing_in_meters = arrow_spacing_in_meters

        # 输出调试信息
        print(f"图片尺寸：{self.pixels_width} x {self.pixels_height} 像素，总像素：{self.pixel_count}")
        print(f"网格总数：{self.x_grid_num * self.y_grid_num}")
        print(f"a = 像素总数/网格总数 = {self.a:.2f}")
        print(f"修正系数 L = (a - 156) = {self.L:.2f}")

        fig_width = self.pixels_width / dpi
        fig_height = self.pixels_height / dpi
        print(f"图幅尺寸（英寸）：{fig_width}, {fig_height}")

        # 创建绘图窗口
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
        """对绿色边界进行平滑和密化处理"""
        boundary_polygon = self.green_border
        bx, by = boundary_polygon.exterior.xy
        points = np.column_stack([bx, by])
        if np.array_equal(points[0], points[-1]):
            points = points[:-1]
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, per=1)
        u_new = np.linspace(0, 1, len(points) * green_edge_sampling_factor, endpoint=False)
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_points = np.column_stack([smooth_x, smooth_y])
        print(f"Smoothed points: {len(points)} -> {len(smooth_points)}")
        return Polygon(smooth_points)

    def _generate_masks(self):
        """生成图像掩膜和高程插值结果"""
        boundary_polygon = self.green_border
        mask = np.zeros_like(self.xi, dtype=bool)
        for i in range(self.xi.shape[0]):
            for j in range(self.xi.shape[1]):
                point = Point(self.xi[i, j], self.yi[i, j])
                mask[i, j] = boundary_polygon.contains(point)
        xi_masked = np.ma.masked_array(self.xi, ~mask)
        yi_masked = np.ma.masked_array(self.yi, ~mask)

        boundary_points = np.array(boundary_polygon.exterior.coords)
        # 对边界点使用高斯卷积插值（卷积插值）
        boundary_z = griddata(self.xys, self.zs, boundary_points, method="linear")

        xys_enhanced = np.vstack([self.xys, boundary_points])
        zs_enhanced = np.hstack([self.zs, boundary_z])

        # 利用高斯卷积插值生成初步的高程图
        zi_linear_contour = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear")
        valid_mask = ~np.isnan(zi_linear_contour)
        if np.any(~valid_mask):
            print(f"发现 {np.sum(~valid_mask)} 个无效点，使用linear插值生成轮廓数据")
            zi_contour = zi_linear_contour
        else:
            zi_contour = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic")
            nan_mask = np.isnan(zi_contour)
            if np.any(nan_mask):
                zi_contour[nan_mask] = zi_linear_contour[nan_mask]
        zi_masked_contour = np.ma.masked_array(zi_contour, ~mask)

        # 生成初步的线性插值高程图
        zi_linear = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="linear")
        nan_mask = np.isnan(zi_linear)
        if np.any(nan_mask):
            print(f"发现 {np.sum(nan_mask)} 个无效点，执行三次卷积插值填充")
            # 使用高斯卷积插值填充NaN值
            zi_linear = gaussian_filter(zi_linear, sigma=3)  # sigma越大平滑效果越明显
        zi_cubic = griddata(xys_enhanced, zs_enhanced, (self.xi, self.yi), method="cubic")
        zi = np.where(np.isnan(zi_cubic), zi_linear, zi_cubic)
        zi_masked = np.ma.masked_array(zi, ~mask)
        return mask, xi_masked, yi_masked, zi_masked, zi_masked_contour

    def _get_arrow_parameters(self):
        """
        计算箭头参数，根据数据特征自适应调整箭头尺寸，
        保证在不同图幅上箭头视觉效果一致。
        """
        total_area = self.width_meters * self.height_meters
        desired_arrow_spacing = self.arrow_spacing_in_meters
        estimated_arrow_count = int(np.sqrt(total_area / (desired_arrow_spacing ** 2)))
        area_normalization_factor = np.log1p(total_area) / 10  # 对数缩放

        min_cell_size = min(self.width_meters / self.x_grid_num, self.height_meters / self.y_grid_num)
        arrow_width = max(0.01, min_cell_size / 100)  # 保证最小可见性

        if self.adj_ratio >= 1.5:
            base_arrow_length_scale = 60
        elif self.adj_ratio >= 0.9:
            base_arrow_length_scale = 50
        else:
            base_arrow_length_scale = 35
        arrow_length_scale = base_arrow_length_scale * area_normalization_factor

        arrow_headwidth = max(6, int(arrow_width * 150))
        arrow_headlength = max(6, int(arrow_width * 150))
        arrow_headaxislength = max(6, int(arrow_width * 150))

        density_factor = np.clip(np.sqrt(estimated_arrow_count) / 10, 0.2, 2.0)
        length_scale = arrow_length_scale * density_factor

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
        """
        计算高程梯度，并进行数值精度处理，
        返回归一化后的梯度分量，用于确定箭头指向。
        """
        epsilon = 1e-8
        gradient_y, gradient_x = np.gradient(zi)
        magnitude = np.hypot(gradient_x, gradient_y)
        magnitude = np.where(magnitude < epsilon, epsilon, magnitude)
        return -gradient_x / magnitude, -gradient_y / magnitude

    def _rectangular_fit_score(self, polygon, threshold=0.8):
        """
        计算多边形与其外接矩形的拟合程度，
        返回拟合得分及是否接近矩形的布尔值。
        """
        minx, miny, maxx, maxy = polygon.bounds
        bounding_box_area = (maxx - minx) * (maxy - miny)

        if bounding_box_area == 0:
            return 0, False  # 防止除零错误

        score = polygon.area / bounding_box_area
        is_rectangular = score >= threshold
        return score, is_rectangular

    def _plot(self):
        """主渲染流程：生成插值图、绘制边界和箭头"""
        mask, xi_masked, yi_masked, zi_masked, zi_masked_contour = self._generate_masks()

        # 绘制高程颜色梯度
        levels = np.linspace(self.zs.min(), self.zs.max(), elevation_levels)
        custom_cmap = colors.LinearSegmentedColormap.from_list("custom", colors_gradient_list)
        self.ax.contourf(xi_masked, yi_masked, zi_masked_contour, levels=levels, cmap=custom_cmap)

        # 绘制绿色边界
        bx, by = self.green_border.exterior.xy
        self.ax.plot(bx, by, color="black", linewidth=3)
        polygon_path = Path(np.column_stack((bx, by)))
        clip_patch = PathPatch(polygon_path, transform=self.ax.transData, facecolor='none', edgecolor='none')

        arrows_params = self._get_arrow_parameters()
        print(f"箭头参数: {arrows_params}")

        # 计算高程梯度，用于箭头的方向
        dx, dy = self._eps_gradient(zi_masked)

        x_grid = np.arange(self.x_min, self.x_max, self.arrow_spacing_in_meters)
        y_grid = np.arange(self.y_min, self.y_max, self.arrow_spacing_in_meters)
        X, Y = np.meshgrid(x_grid, y_grid)

        # 将规则网格展开
        xi_flat = self.xi.flatten()
        yi_flat = self.yi.flatten()

        # 在箭头网格上插值梯度分量
        U = griddata((xi_flat, yi_flat), dx.flatten(), (X, Y), method='linear', fill_value=0)
        V = griddata((xi_flat, yi_flat), dy.flatten(), (X, Y), method='linear', fill_value=0)

        # 归一化梯度向量
        magnitude = np.sqrt(U ** 2 + V ** 2)
        eps = 1e-8
        U = U / (magnitude + eps)
        V = V / (magnitude + eps)

        # 根据绿色边界确定箭头绘制区域
        score, is_rectangular = self._rectangular_fit_score(self.green_border)
        buffer_length = -0.6 if is_rectangular else -1.665
        buffered = self.green_border.buffer(buffer_length)
        valid = np.array([buffered.covers(Point(x, y)) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

        # 调整箭头长度参数，考虑修正系数
        arrows_params['length_scale'] *= (1 + self.L / 100)

        # 绘制箭头
        quiver_collection = self.ax.quiver(
            X[valid], Y[valid], U[valid], V[valid],
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
        self.arrow_count = len(quiver_collection.get_offsets())

        # 保存图像
        plt.savefig(self.output_path, bbox_inches="tight", pad_inches=0, transparent=True, dpi=dpi)
        plt.close()

    def process_file(self, json_path, output_path):
        """文件处理流水线：加载数据、初始化、绘制并输出统计信息"""
        self._reset()
        self.output_path = output_path
        self.data = self._load_json(json_path)
        self._init()
        self._plot()

        # 生成并输出统计信息
        image_area = self.width_meters * self.height_meters
        arrow_density = self.arrow_count / image_area if image_area > 0 else 0
        print("\n" + "=" * 60)
        print(f"文件: {os.path.basename(output_path)}")
        print(f"像素尺寸: {self.pixels_width} x {self.pixels_height} (宽x高)")
        print(f"地理尺寸: {self.width_meters:.2f}m x {self.height_meters:.2f}m")
        print(f"箭头总数: {self.arrow_count}")
        print(f"地理密度: {(1 / arrow_density):.2f} arrows/sqm")
        print(f"像素密度: {1 / (self.arrow_count / self.pixel_count)} arrows/pixel")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    input_dir = r"C:\Users\13211\Desktop\upwork\map_first\green_visualizer-main\testcases\json"
    output_dir = input_dir.replace("json", "map")

    visualizer = GreenVisualizer()
    for case_id in range(1, 19):
        input_file = os.path.join(input_dir, f"{case_id}.json")
        output_file = os.path.join(output_dir, f"{case_id}.png")
        visualizer.process_file(input_file, output_file)