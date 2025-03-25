"""
地理数据可视化系统
功能：读取包含高程点和绿地边界的GeoJSON数据，生成带等高线、梯度箭头和边界轮廓的地形图
特点：基于最小外接矩形(MBR)的地理校正、抗锯齿渲染、自适应网格生成
"""

import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata, RegularGridInterpolator
import numpy as np
from matplotlib.path import Path
import os
from scipy.ndimage import gaussian_filter

# -------------------------- 系统级配置参数 --------------------------
# 坐标系参数
dpi = 300  # 图像输出分辨率（每英寸点数）
lat_to_meter_ratio = 111000  # 纬度转米系数（1度≈111公里）
target_meters_per_pixel = 0.01  # 目标分辨率（米/像素）
buffer_distance = 1e-6  # 多边形缓冲距离（单位：度）

# 地形渲染配置
base_grid_num = 800  # 基础网格划分密度
gaussian_sigma = 1.8  # 高斯模糊强度（值越大越平滑）
color_levels = 16  # 颜色分层数量
contour_linewidth = 0.8  # 等高线线宽（单位：点）

# 箭头可视化配置
arrow_count = 15  # 每个方向箭头数量
arrow_scale = 20  # 箭头尺寸缩放系数
arrow_width = 0.004  # 箭头杆宽度（相对图像尺寸）
arrow_headwidth = 5  # 箭头头部宽度倍数
arrow_headlength = 6  # 箭头头部长度倍数
arrow_alpha = 0.85  # 箭头透明度（0-1）
min_magnitude = 1e-8  # 梯度计算最小阈值

# 图形样式配置
outline_width = 0.8  # 边界轮廓线宽（单位：点）

# 高程色谱配置（保持英文颜色注释）
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

# 配置中文字体支持（不需要）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


class GreenVisualizer:
    """地理可视化核心处理器，封装数据加载、处理、渲染全流程"""

    def __init__(self):
        """初始化实例状态"""
        self._reset()

    def _reset(self):
        """重置所有实例变量"""
        self.data = None  # 原始GeoJSON数据
        self.elevation_points = []  # 高程点三维坐标列表
        self.green_border = None  # 绿地边界多边形对象
        self.output_path = None  # 输出文件路径
        self.xys = None  # 高程点二维坐标矩阵
        self.zs = None  # 高程值数组
        self.xi, self.yi = None, None  # 插值网格坐标矩阵
        self.x_min = self.x_max = None  # 地理范围X轴边界
        self.y_min = self.y_max = None  # 地理范围Y轴边界
        self.ax, self.fig = None, None  # Matplotlib绘图对象
        self.mask = None  # 有效区域掩膜矩阵

    @staticmethod
    def _load_json(json_path):
        """加载并验证GeoJSON文件"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _geo_correction(self):
        """执行地理坐标系校正"""
        # 计算最小外接矩形
        minx, miny, maxx, maxy = self.green_border.bounds
        self.x_min, self.x_max = minx, maxx
        self.y_min, self.y_max = miny, maxy

        # 基于中心纬度计算米像素比
        y_center = (self.y_min + self.y_max) / 2
        lat_rad = np.radians(y_center)

        # 转换地理范围到米制单位
        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min
        width_m = x_span * lat_to_meter_ratio * np.cos(lat_rad)
        height_m = y_span * lat_to_meter_ratio

        # 计算网格尺寸（限制最大10000像素）
        pixels_x = int(width_m / target_meters_per_pixel)
        pixels_y = int(height_m / target_meters_per_pixel)
        return min(pixels_x, 10000), min(pixels_y, 10000)

    def _init_grid(self):
        """初始化插值网格和绘图画布"""
        grid_w, grid_h = self._geo_correction()

        # 创建均匀分布的地理网格
        self.xi = np.linspace(self.x_min, self.x_max, grid_w)
        self.yi = np.linspace(self.y_min, self.y_max, grid_h)
        self.xi, self.yi = np.meshgrid(self.xi, self.yi)

        # 配置高精度地理绘图参数
        self.fig, self.ax = plt.subplots(
            figsize=(grid_w / dpi, grid_h / dpi), dpi=dpi, facecolor="none"
        )
        self.ax.axis("off")
        self.ax.set_aspect("equal")
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)

    def _init(self):
        """主初始化流程"""
        # 解析GeoJSON特征数据
        for feature in self.data["features"]:
            if feature["id"] == "Elevation":
                geom = feature["geometry"]
                # 处理多点/单点数据结构
                if geom["type"] == "MultiPoint":
                    self.elevation_points.extend(geom["coordinates"])
                else:
                    self.elevation_points.append(geom["coordinates"])
            elif feature["id"] == "GreenBorder":
                self.green_border = Polygon(feature["geometry"]["coordinates"])

        # 转换坐标数据为NumPy数组
        self.xys = np.array([[p[0], p[1]] for p in self.elevation_points])
        self.zs = np.array([p[2] for p in self.elevation_points])

        # 执行地理校正和网格初始化
        self._geo_correction()
        self._init_grid()

    def _generate_masks(self):
        """生成地形掩膜和高程插值数据"""
        # 创建缓冲多边形路径
        buffered = self.green_border.buffer(buffer_distance)
        polygon_path = Path(buffered.exterior.coords)

        # 生成有效区域掩膜
        grid_points = np.column_stack([self.xi.ravel(), self.yi.ravel()])
        mask = polygon_path.contains_points(grid_points).reshape(self.xi.shape)

        # 边界点加密处理
        original_edges = np.array(buffered.exterior.coords)
        dense_edges = np.vstack(
            [
                np.linspace(
                    original_edges[i], original_edges[i + 1], len(original_edges) * 3
                )
                for i in range(len(original_edges) - 1)
            ]
        )
        edge_elevations = griddata(self.xys, self.zs, dense_edges, method="nearest")

        # 合并原始点和加密点
        full_coords = np.vstack([self.xys, dense_edges])
        full_values = np.concatenate([self.zs, edge_elevations])

        # 混合插值算法（线性+高斯滤波）
        zi = griddata(full_coords, full_values, (self.xi, self.yi), method="linear")
        zi = gaussian_filter(zi, sigma=gaussian_sigma)

        # 处理残留NaN值
        zi_nearest = griddata(
            full_coords, full_values, (self.xi, self.yi), method="nearest"
        )
        zi[np.isnan(zi) & mask] = zi_nearest[np.isnan(zi) & mask]
        zi[~mask] = np.nan

        return mask, zi

    def _safe_gradient(self, zi):
        """安全梯度计算（防止除零错误）"""
        dy, dx = np.gradient(zi)
        mag = np.hypot(dx, dy)
        mag[mag < min_magnitude] = min_magnitude  # 设置最小值阈值
        return -dx / mag, -dy / mag  # 返回负梯度（下坡方向）

    def _render_arrows(self, zi):
        """梯度箭头可视化引擎"""
        dx, dy = self._safe_gradient(zi)

        # 生成箭头采样网格
        y_idx = np.linspace(0, self.xi.shape[0] - 1, arrow_count, dtype=int)
        x_idx = np.linspace(0, self.xi.shape[1] - 1, arrow_count, dtype=int)

        X = self.xi[np.ix_(y_idx, x_idx)]
        Y = self.yi[np.ix_(y_idx, x_idx)]
        U = dx[np.ix_(y_idx, x_idx)]
        V = dy[np.ix_(y_idx, x_idx)]

        # 验证箭头位置有效性
        buffered = self.green_border.buffer(buffer_distance)
        path = Path(buffered.exterior.coords)
        valid = path.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(
            X.shape
        )

        # 边界容差处理
        edge_tolerance = 1e-8  # 坐标容差阈值
        edge_mask = (
            (np.abs(X - self.x_min) < edge_tolerance)
            | (np.abs(X - self.x_max) < edge_tolerance)
            | (np.abs(Y - self.y_min) < edge_tolerance)
            | (np.abs(Y - self.y_max) < edge_tolerance)
        )
        valid |= edge_mask

        # 绘制有效箭头
        self.ax.quiver(
            X[valid],
            Y[valid],
            U[valid],
            V[valid],
            color="black",
            scale=arrow_scale,
            width=arrow_width,
            headwidth=arrow_headwidth,
            headlength=arrow_headlength,
            minshaft=2,
            alpha=arrow_alpha,
            zorder=15,
            pivot="middle",
        )

    def _add_outline(self):
        """添加抗锯齿边界轮廓"""
        if self.mask is None:
            return

        # 生成轮廓线并设置样式
        mask_float = self.mask.astype(float)
        contours = self.ax.contour(
            self.xi,
            self.yi,
            mask_float,
            levels=[0.5],
            colors="black",
            linewidths=outline_width,
        )

        # 设置线条连接样式
        for contour in contours.collections:
            contour.set_capstyle("round")
            contour.set_joinstyle("round")
            contour.set_zorder(20)

    def _plot(self):
        """主绘图流程"""
        # 生成地形数据和掩膜
        self.mask, zi = self._generate_masks()
        zi_masked = np.ma.masked_array(zi, ~self.mask)

        # 设置颜色分级
        levels = np.linspace(self.zs.min(), self.zs.max(), color_levels)
        cmap = colors.ListedColormap(colors_gradient_list)

        # 绘制填充等高线
        self.ax.contourf(self.xi, self.yi, zi_masked, levels=levels, cmap=cmap)

        # 添加等高线（根据最新需求，不再需要）
        # self.ax.contour(
        #     self.xi, self.yi, zi_masked,
        #     levels=levels,
        #     linewidths=contour_linewidth,
        #     colors='black',
        #     antialiased=True
        # )

        # 添加可视化元素
        self._render_arrows(zi_masked)
        self._add_outline()

        # 添加颜色图例(不需要）
        # norm = colors.BoundaryNorm(levels, len(colors_gradient_list))
        # cbar = plt.colorbar(
        #     plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        #     ax=self.ax,
        #     orientation='vertical',
        #     fraction=0.03,
        #     pad=0.04
        # )
        # cbar.set_label('海拔高度 (米)', fontsize=10)

    def process_file(self, json_path, output_path):
        """处理单个文件的全流程"""
        self._reset()
        try:
            # 数据加载与初始化
            self.data = self._load_json(json_path)
            self._init()

            # 创建输出目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 执行绘图并保存
            self._plot()
            plt.savefig(
                output_path,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=dpi,
            )
            print(f"成功生成: {output_path}")
        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {str(e)}")
        finally:
            plt.close("all")


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
