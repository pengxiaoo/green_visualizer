import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata, RegularGridInterpolator
import numpy as np
from matplotlib.path import Path
import os
from scipy.ndimage import gaussian_filter
from matplotlib.patches import PathPatch

# -------------------------- 系统级配置参数 --------------------------
dpi = 300
lat_to_meter_ratio = 111000       # 每纬度度数对应的米数
target_meters_per_pixel = 0.01    # 每像素代表的米数（用于尺寸转换）
buffer_distance = 1e-6

# 地形渲染配置
base_grid_num = 800
gaussian_sigma = 1.8
color_levels = 16

# 箭头系统参数（动态计算间距）
arrow_scale = 18
arrow_width = 0.008
arrow_headwidth = 5
arrow_headlength = 6
arrow_alpha = 0.92
min_magnitude = 1e-8
grid_spacing_pixels = 475  # 每个箭头间隔150像素

# 轮廓线参数
outline_width = 0.8
outline_buffer = 1.5
contour_level = 0.5

# 高程色谱配置
colors_gradient_list = [
    "#1640C5", "#126ED4", "#1C9AD9", "#0BBBCA",
    "#1AD7C6", "#3ADE8A", "#4FE670", "#9AE639",
    "#E1CF24", "#E5A129", "#E8862A", "#E36626",
    "#F2451D", "#EF4123", "#EB3B2A", "#CA253C"
]


class GreenVisualizer:
    """地理可视化处理器（增强版）"""

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
        self.xi, self.yi = None, None
        self.x_min = self.x_max = None
        self.y_min = self.y_max = None
        self.ax, self.fig = None, None
        self.mask = None
        self.clip_path = None

    @staticmethod
    def _load_json(json_path):
        """安全加载GeoJSON文件"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"文件未找到: {json_path}")
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _geo_correction(self):
        """执行地理坐标校正"""
        minx, miny, maxx, maxy = self.green_border.bounds
        self.x_min, self.x_max = minx, maxx
        self.y_min, self.y_max = miny, maxy

        y_center = (self.y_min + self.y_max) / 2
        lat_rad = np.radians(y_center)

        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min
        width_m = x_span * lat_to_meter_ratio * np.cos(lat_rad)
        height_m = y_span * lat_to_meter_ratio

        # 返回图像所需的网格大小，并动态调整以适应不同地理坐标范围和图像尺寸
        return (
            min(int(width_m / target_meters_per_pixel), 10000),
            min(int(height_m / target_meters_per_pixel), 10000)
        )

    def _init_grid(self):
        """初始化插值网格系统"""
        grid_w, grid_h = self._geo_correction()

        self.xi = np.linspace(self.x_min, self.x_max, grid_w)
        self.yi = np.linspace(self.y_min, self.y_max, grid_h)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

        self.fig, self.ax = plt.subplots(
            figsize=(grid_w / dpi, grid_h / dpi),
            dpi=dpi,
            facecolor="none"
        )
        self.ax.axis('off')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

    def _init(self):
        """主初始化流程"""
        for feature in self.data["features"]:
            if feature["id"] == "Elevation":
                geom = feature["geometry"]
                if geom["type"] == "MultiPoint":
                    self.elevation_points.extend(geom["coordinates"])
                else:
                    self.elevation_points.append(geom["coordinates"])
            elif feature["id"] == "GreenBorder":
                self.green_border = Polygon(feature["geometry"]["coordinates"])

        self.xys = np.array([[p[0], p[1]] for p in self.elevation_points])
        self.zs = np.array([p[2] for p in self.elevation_points])

        self._geo_correction()
        self._init_grid()

    def _generate_masks(self):
        """生成地形掩膜系统"""
        buffered = self.green_border.buffer(buffer_distance)
        polygon_path = Path(buffered.exterior.coords)
        self.clip_path = polygon_path

        grid_points = np.column_stack([self.xi.ravel(), self.yi.ravel()])
        mask = polygon_path.contains_points(grid_points).reshape(self.xi.shape)

        original_edges = np.array(buffered.exterior.coords)
        dense_edges = np.vstack([
            np.linspace(original_edges[i], original_edges[i + 1], len(original_edges) * 3)
            for i in range(len(original_edges) - 1)
        ])
        edge_elevations = griddata(self.xys, self.zs, dense_edges, method="nearest")

        full_coords = np.vstack([self.xys, dense_edges])
        full_values = np.concatenate([self.zs, edge_elevations])

        zi = griddata(full_coords, full_values, (self.xi, self.yi), method="linear")
        zi = gaussian_filter(zi, sigma=gaussian_sigma)

        zi_nearest = griddata(full_coords, full_values, (self.xi, self.yi), method="nearest")
        zi[np.isnan(zi) & mask] = zi_nearest[np.isnan(zi) & mask]
        zi[~mask] = np.nan

        return mask, zi

    def _safe_gradient(self, zi):
        """安全梯度计算"""
        dy, dx = np.gradient(zi)
        mag = np.hypot(dx, dy)
        mag[mag < min_magnitude] = min_magnitude
        return -dx / mag, -dy / mag

    def _render_arrows(self, zi):
        """网格化箭头生成引擎（增强版）"""
        dx, dy = self._safe_gradient(zi)

        # 根据目标像素间隔计算实际物理间隔（单位：米）
        grid_spacing_meters = grid_spacing_pixels * target_meters_per_pixel

        # 计算当前地图中心纬度（用于经度距离转换）
        lat_center = (self.y_min + self.y_max) / 2
        cos_lat = np.cos(np.radians(lat_center))

        # 将物理间隔转换为经纬度单位（注意：纬度和经度的转换比例不同）
        delta_lon = grid_spacing_meters / (lat_to_meter_ratio * cos_lat)
        delta_lat = grid_spacing_meters / lat_to_meter_ratio

        # 使用统一的间隔确保箭头在 X 和 Y 方向上均匀分布
        delta = min(delta_lon, delta_lat)  # 统一箭头间距，避免重叠

        # 生成网格时保证箭头位于中心区域
        x_coords = np.arange(self.x_min + delta / 2, self.x_max, delta)
        y_coords = np.arange(self.y_min + delta / 2, self.y_max, delta)
        X, Y = np.meshgrid(x_coords, y_coords)

        # 创建插值器，将梯度值从整体网格插值到箭头网格上
        x_orig = self.xi[0, :]
        y_orig = self.yi[:, 0]
        interp_dx = RegularGridInterpolator((y_orig, x_orig), dx)
        interp_dy = RegularGridInterpolator((y_orig, x_orig), dy)
        points = np.column_stack((Y.ravel(), X.ravel()))
        U = interp_dx(points).reshape(X.shape)
        V = interp_dy(points).reshape(X.shape)

        # 创建边界掩膜，确保箭头只在有效区域内显示
        buffered = self.green_border.buffer(buffer_distance)
        path = Path(buffered.exterior.coords)
        valid = path.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        valid &= ~np.isnan(U) & ~np.isnan(V)

        # 创建裁剪路径，防止箭头绘制超出边界
        clip_patch = PathPatch(self.clip_path, transform=self.ax.transData, visible=False)
        self.ax.add_patch(clip_patch)

        # 绘制箭头，使用固定的 scale 参数确保大小一致
        quiver = self.ax.quiver(
            X[valid], Y[valid], U[valid], V[valid],
            color='black',
            scale=arrow_scale,
            width=arrow_width,
            headwidth=arrow_headwidth,
            headlength=arrow_headlength,
            minshaft=2,
            alpha=arrow_alpha,
            zorder=15,
            pivot='middle'
        )
        quiver.set_clip_path(clip_patch)

    def _add_outline(self):
        """双轮廓线生成系统"""
        if self.mask is None:
            return

        mask_float = self.mask.astype(float)

        # 增加轮廓线缓冲，确保边界区域被完整绘制
        contours_black = self.ax.contour(
            self.xi, self.yi, mask_float,
            levels=[contour_level],
            colors='black',
            linewidths=outline_width + outline_buffer,
            linestyles='solid',
            zorder=20
        )

        contours_white = self.ax.contour(
            self.xi, self.yi, mask_float,
            levels=[contour_level],
            colors='black',
            linewidths=outline_width,
            linestyles='solid',
            zorder=21
        )

        for contour in contours_black.collections:
            contour.set_antialiased(True)
        for contour in contours_white.collections:
            contour.set_antialiased(True)

    def _plot(self):
        """主渲染流程"""
        self.mask, zi = self._generate_masks()
        zi_masked = np.ma.masked_array(zi, ~self.mask)

        levels = np.linspace(self.zs.min(), self.zs.max(), color_levels)
        cmap = colors.ListedColormap(colors_gradient_list)
        self.ax.contourf(self.xi, self.yi, zi_masked, levels=levels, cmap=cmap, zorder=10)

        self._render_arrows(zi_masked)
        self._add_outline()

    def process_file(self, json_path, output_path):
        """文件处理流水线"""
        self._reset()
        try:
            self.data = self._load_json(json_path)
            self._init()

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            self._plot()
            plt.savefig(
                output_path,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True,
                dpi=dpi
            )
            print(f"输出成功: {output_path}")
        except Exception as e:
            print(f"处理错误 {json_path}: {str(e)}")
        finally:
            plt.close('all')


if __name__ == "__main__":
    input_dir = r"C:\Users\13211\Desktop\upwork\map_first\green_visualizer-main\testcases\json"
    output_dir = input_dir.replace("json", "map")

    visualizer = GreenVisualizer()
    for case_id in range(1, 19):
        input_file = os.path.join(input_dir, f"{case_id}.json")
        output_file = os.path.join(output_dir, f"{case_id}.png")
        visualizer.process_file(input_file, output_file)