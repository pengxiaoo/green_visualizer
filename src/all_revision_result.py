import json
import os
import logging
from math import sqrt
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator, splprep, splev
from pyproj import Transformer, CRS

# -------------------------- 系统级配置参数 --------------------------
input_crs = CRS.from_string('EPSG:4326')
output_crs = CRS.from_string('EPSG:3857')
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

dpi = 300
target_meters_per_pixel = 0.02
base_grid_num = 400
arrow_spacing_in_meters = 6
green_edge_sampling_factor = 3
grid_density = 16

colors_gradient_list = [
    "#1640C5", "#126ED4", "#1C9AD9", "#0BBBCA", "#1AD7C6",
    "#3ADE8A", "#4FE670", "#9AE639", "#E1CF24", "#E5A129",
    "#E8862A", "#E36626", "#F2451D", "#EF4123", "#EB3B2A", "#CA253C"
]
elevation_levels = len(colors_gradient_list)


# -------------------------- 图像处理配置 --------------------------
class ImageConfig:
    """图像处理配置参数"""
    LOWER_BLACK = np.array([0, 0, 0])
    UPPER_BLACK = np.array([180, 50, 30])
    MAX_ITERATIONS = 50
    CONVERGENCE_THRESHOLD = 0.01
    MIN_IMAGE_SIZE = 100
    MORPH_PERCENT = 0.005
    CONTOUR_THICKNESS = 1
    EDGE_PADDING = 2


# -------------------------- 高程可视化类 --------------------------
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

    def _fill_nan_bilinear(self, data, max_iter=10):
        """
        使用迭代法进行双线性插值填充NaN值：
        对于每个缺失值，利用其上下左右相邻点的有效值的平均值来填补，
        重复迭代直到所有缺失值填补完毕或达到最大迭代次数。
        """
        filled = data.copy()
        for _ in range(max_iter):
            nan_mask = np.isnan(filled)
            if not np.any(nan_mask):
                break
            filled_new = filled.copy()
            # 利用np.roll获取上下左右相邻的数据
            neighbors = []
            for shift in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbors.append(np.roll(filled, shift, axis=(0,1)))
            neighbors = np.array(neighbors)  # shape: (4, n, m)
            # 计算每个位置的有效邻居的和与个数
            valid = ~np.isnan(neighbors)
            sum_neighbors = np.nansum(neighbors, axis=0)
            count_neighbors = np.sum(valid, axis=0)
            # 对于缺失值且至少有一个有效邻居的点，填充平均值
            update = np.where((nan_mask) & (count_neighbors > 0), sum_neighbors / count_neighbors, filled)
            filled_new[nan_mask] = update[nan_mask]
            filled = filled_new
        return filled

    def _generate_masks(self):
        """生成图像掩膜和高程插值结果（完整修复版）"""
        boundary_polygon = self.green_border

        # ====================================================================
        # Step 1: 高精度掩膜生成（修复坐标遍历逻辑）
        # ====================================================================
        buffered_polygon = boundary_polygon.buffer(0.1)  # 0.1米缓冲防止边缘漏点
        mask = np.zeros_like(self.xi, dtype=bool)

        # 关键修复：使用双重循环逐个处理网格点
        for i in range(self.xi.shape[0]):
            for j in range(self.xi.shape[1]):
                # 确保坐标是标量值
                x = self.xi[i, j].item()  # 转换为Python标量
                y = self.yi[i, j].item()
                mask[i, j] = buffered_polygon.contains(Point(x, y))

        # ====================================================================
        # Step 2: 边界数据增强（确保边界点高程有效性）
        # ====================================================================
        boundary_points = np.array(boundary_polygon.exterior.coords)

        # 双重保障边界点高程计算
        nearest_interp = NearestNDInterpolator(self.xys, self.zs)
        boundary_z_nearest = nearest_interp(*boundary_points.T)  # 最近邻估算

        boundary_z_linear = griddata(
            self.xys, self.zs,
            boundary_points, method="linear"  # 线性插值优化
        )
        # 融合两种插值结果
        boundary_z = np.where(
            np.isnan(boundary_z_linear),
            boundary_z_nearest,
            (boundary_z_linear * 0.7 + boundary_z_nearest * 0.3)  # 加权混合
        )

        # 构建增强数据集
        xys_enhanced = np.vstack([self.xys, boundary_points])
        zs_enhanced = np.hstack([self.zs, boundary_z])

        # ====================================================================
        # Step 3: 主插值（双线性）
        # ====================================================================
        zi = griddata(
            xys_enhanced, zs_enhanced,
            (self.xi, self.yi), method="linear"
        )

        # ====================================================================
        # Step 4: 迭代填充（仅处理轮廓线内区域）
        # ====================================================================
        nan_mask = np.isnan(zi) & mask
        for _ in range(3):  # 有限迭代3次
            if not np.any(nan_mask):
                break
            # 构建局部插值器（仅使用有效数据）
            valid_points = np.column_stack([
                self.xi[~np.isnan(zi) & mask],
                self.yi[~np.isnan(zi) & mask]
            ])
            valid_values = zi[~np.isnan(zi) & mask]

            if len(valid_points) > 0:
                interp = LinearNDInterpolator(valid_points, valid_values)
                zi[nan_mask] = interp(
                    self.xi[nan_mask],
                    self.yi[nan_mask]
                )
            nan_mask = np.isnan(zi) & mask

        # ====================================================================
        # Step 5: 最近邻兜底填充（100%覆盖轮廓线内区域）
        # ====================================================================
        if np.any(nan_mask):
            print(f"最终兜底填充：{np.sum(nan_mask)}个点")
            # 仅使用轮廓线内已知有效点
            known_points = np.column_stack([
                self.xi[mask & ~np.isnan(zi)],
                self.yi[mask & ~np.isnan(zi)]
            ])
            known_values = zi[mask & ~np.isnan(zi)]

            if len(known_points) > 0:
                nearest_interp = NearestNDInterpolator(known_points, known_values)
                zi[nan_mask] = nearest_interp(
                    self.xi[nan_mask],
                    self.yi[nan_mask]
                )
            else:
                # 极端情况：所有点都无效时填充固定值
                zi[nan_mask] = np.nanmean(zs_enhanced)

        # ====================================================================
        # Step 6: 数据裁剪与输出
        # ====================================================================
        zi_masked = np.ma.masked_array(zi, ~mask)
        zi_masked_contour = np.ma.masked_array(zi, ~mask)  # 保持接口兼容

        return mask, self.xi, self.yi, zi_masked, zi_masked_contour

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
            return 0, False
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


# -------------------------- 箭头处理器类 --------------------------
class ArrowProcessor:
    """图像后处理器（尺寸标准化和锐化增强）"""

    def __init__(self, reference_path):
        self.cfg = ImageConfig()
        self.target_size = self._load_reference(reference_path)
        logging.info(f"基准尺寸：{self.target_size}")

    def _load_reference(self, path):
        """加载并验证基准图像"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError("无法读取基准图像")
        mask = self._create_mask(img)
        contour = self._find_main_contour(mask)
        if contour is None:
            raise ValueError("基准图像未检测到有效轮廓")
        w, h = cv2.boundingRect(contour)[2:]
        return (w, h)

    def _create_mask(self, img):
        """创建保护轮廓的掩膜"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.cfg.LOWER_BLACK, self.cfg.UPPER_BLACK)
        kernel = self._dynamic_kernel(img)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _dynamic_kernel(self, img):
        """动态计算形态学核尺寸"""
        h, w = img.shape[:2]
        size = int(min(h, w) * self.cfg.MORPH_PERCENT)
        size = max(3, size | 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _find_main_contour(self, mask):
        """寻找最大轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    def _calculate_optimal_scale(self, contour):
        """计算最佳缩放比例，最大不超过5倍"""
        x, y, w, h = cv2.boundingRect(contour)
        target_w, target_h = self.target_size
        scale = sqrt((target_w / w) * (target_h / h))
        return min(scale, 3)

    def _smart_scale(self, img, scale):
        """智能缩放（保护边缘）"""
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        channels = [cv2.resize(c, new_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                    for c in cv2.split(img)]
        scaled = cv2.merge(channels)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]]) / 4.0
        return cv2.filter2D(scaled, -1, sharpen_kernel)

    def _restore_contours(self, scaled_img, original_mask):
        """轮廓修复"""
        edges = cv2.Canny(original_mask, 50, 150)
        scaled_edges = cv2.resize(edges, (scaled_img.shape[1], scaled_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((self.cfg.CONTOUR_THICKNESS, self.cfg.CONTOUR_THICKNESS), np.uint8)
        scaled_edges = cv2.dilate(scaled_edges, kernel)
        scaled_img[scaled_edges > 0] = [0, 0, 0]
        return scaled_img

    def process_image(self, input_path, output_dir):
        """处理单张图像"""
        try:
            img = cv2.imread(input_path)
            original_mask = self._create_mask(img)
            main_contour = self._find_main_contour(original_mask)
            scale = self._calculate_optimal_scale(main_contour)
            scaled_img = self._smart_scale(img, scale)
            final_img = self._restore_contours(scaled_img, original_mask)
            output_path = os.path.join(output_dir, os.path.basename(input_path))
            cv2.imwrite(output_path, final_img)
            return True
        except Exception as e:
            logging.error(f"处理 {input_path} 失败：{str(e)}")
            return False


# -------------------------- 主流程控制 --------------------------
def generate_elevation_maps():
    """生成高程图"""
    input_dir = r"C:\Users\13211\Desktop\upwork\map_first\green_visualizer-main\testcases\json"
    output_dir = input_dir.replace("json", "map")
    os.makedirs(output_dir, exist_ok=True)

    visualizer = GreenVisualizer()
    for case_id in range(1, 19):
        input_file = os.path.join(input_dir, f"{case_id}.json")
        output_file = os.path.join(output_dir, f"{case_id}.png")
        visualizer.process_file(input_file, output_file)
    return output_dir


def process_images(map_dir):
    """处理生成的高程图"""
    output_dir = map_dir.replace("map", "map")
    os.makedirs(output_dir, exist_ok=True)
    reference_path = os.path.join(map_dir, "1.png")

    processor = ArrowProcessor(reference_path)
    files = sorted([f for f in os.listdir(map_dir) if f.endswith(".png")],
                   key=lambda x: int(x.split('.')[0]))

    with ThreadPoolExecutor() as executor:
        futures = {}
        for filename in files:
            if filename == "1.png":
                continue  # 跳过基准图
            input_path = os.path.join(map_dir, filename)
            futures[filename] = executor.submit(processor.process_image, input_path, output_dir)

        # 直接复制基准图
        base_src = os.path.join(map_dir, "1.png")
        base_dst = os.path.join(output_dir, "1.png")
        cv2.imwrite(base_dst, cv2.imread(base_src))
        print(f"[1/18] 1.png 基准图已复制")

        # 处理并打印结果
        for idx in range(2, 19):
            filename = f"{idx}.png"
            future = futures.get(filename)
            try:
                success = future.result()
                status = "成功" if success else "失败"
                print(f"[{idx}/18] {filename} 处理{status}")
            except Exception as e:
                print(f"[{idx}/18] {filename} 异常：{str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    # 第一步：生成所有高程图
    map_dir = generate_elevation_maps()

    # 第二步：处理生成的高程图
    process_images(map_dir)