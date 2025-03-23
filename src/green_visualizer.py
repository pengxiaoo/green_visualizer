import json
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import numpy as np

dpi = 200
target_meters_per_pixel = 0.02
lat_to_meter_ratio = 111000
base_grid_num = 120
## todo: the following arrow settings are not good enough
arrow_padding = 1
arrow_count = 8
arrow_interval_min = 4
arrow_interval_max = 10
arrow_length_scale_base = 20
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
        self.ax = None

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
        self.ax = None

    @staticmethod
    def _load_json(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    def _init(self):
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

        # Convert point data to numpy arrays
        self.xys = np.array([[p["x"], p["y"]] for p in self.elevation_points])
        self.zs = np.array([p["z"] for p in self.elevation_points])
        self.x_min, self.x_max = min(self.xys[:, 0]), max(self.xys[:, 0])
        self.y_min, self.y_max = min(self.xys[:, 1]), max(self.xys[:, 1])

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
        width_meters = self.x_range * lat_to_meter_ratio * np.cos(center_lat_rad)
        height_meters = self.y_range * lat_to_meter_ratio

        # 计算需要的像素数
        pixels_width = int(width_meters / target_meters_per_pixel)
        pixels_height = int(height_meters / target_meters_per_pixel)

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
        boundary_points = np.column_stack([bx, by])

        # 计算相邻点之间的距离
        distances = np.sqrt(np.sum(np.diff(boundary_points, axis=0) ** 2, axis=1))
        d_avg = np.mean(distances)

        # 基于平均距离进行插值
        dense_points = []
        for i in range(len(boundary_points)):
            p1 = boundary_points[i]
            p2 = boundary_points[(i + 1) % len(boundary_points)]  # 循环到第一个点

            dense_points.append(p1)
            d = np.sqrt(np.sum((p2 - p1) ** 2))

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
        smooth_polygon = self._smooth_and_densify_edge()
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

    def _plot(self):
        # 生成掩码和插值结果
        mask, xi_masked, yi_masked, zi_masked = self._generate_masks()

        # todo: the edge is not smooth enough
        # Paint the color gradient
        levels = np.linspace(self.zs.min(), self.zs.max(), elevation_levels)
        custom_cmap = colors.LinearSegmentedColormap.from_list(
            "custom", colors_gradient_list
        )
        self.ax.contourf(
            xi_masked, yi_masked, zi_masked, levels=levels, cmap=custom_cmap
        )

        # Paint contour lines
        self.ax.contour(
            xi_masked, yi_masked, zi_masked, levels=levels, colors="k", alpha=0.1
        )

        # Paint gradient arrows
        dx, dy = np.gradient(zi_masked)
        x_spacing = self.x_range / self.x_grid_num
        y_spacing = self.y_range / self.y_grid_num
        dx = dx / x_spacing
        dy = dy / y_spacing
        magnitude = np.sqrt(dx**2 + dy**2)
        dx_normalized = dx / magnitude
        dy_normalized = dy / magnitude
        x_arrow_interval = int(self.x_grid_num / arrow_count)
        x_arrow_interval = max(x_arrow_interval, arrow_interval_min)
        x_arrow_interval = min(x_arrow_interval, arrow_interval_max)
        y_arrow_interval = int(self.y_grid_num / arrow_count)
        y_arrow_interval = max(y_arrow_interval, arrow_interval_min)
        y_arrow_interval = min(y_arrow_interval, arrow_interval_max)
        json_file_index = self.output_path.split("/")[-1].split(".")[0]
        print(
            f"json_file_index: {json_file_index} => x_grid_num: {self.x_grid_num}, y_grid_num: {self.y_grid_num}, "
            f"x_arrow_interval: {x_arrow_interval}, y_arrow_interval: {y_arrow_interval}"
        )
        skip = (
            slice(arrow_padding, -arrow_padding, y_arrow_interval),
            slice(arrow_padding, -arrow_padding, x_arrow_interval),
        )
        mask_skip = mask[skip]
        diagonal_grid_num = np.sqrt(self.x_grid_num**2 + self.y_grid_num**2)
        arrow_length_scale = (
            arrow_length_scale_base * self.x_grid_num / diagonal_grid_num
        )
        print(
            f"diagonal_grid_num: {diagonal_grid_num}, arrow length scale: {arrow_length_scale}"
        )
        self.ax.quiver(
            xi_masked[skip][mask_skip],
            yi_masked[skip][mask_skip],
            -dy_normalized[skip][mask_skip],
            -dx_normalized[skip][mask_skip],
            scale=arrow_length_scale,
            scale_units="width",
            units="width",
            width=0.005,
            headwidth=6,
            headlength=6,
            headaxislength=4,
            minshaft=1,
            minlength=3,
            color="white",
            alpha=1,
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
