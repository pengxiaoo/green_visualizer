import numpy as np
import trimesh
from scipy.spatial import Delaunay
from green_visualizer_2d import GreenVisualizer2D

class GreenVisualizer3D(GreenVisualizer2D):
    def _plot(self, export_path="green_3d.glb"):
        intersection_polygon = self.green_border.intersection(self.polygon)
        if intersection_polygon.is_empty:
            raise Exception(
                f"hole {self.hole_number} green boundary 与 xys 没有相交区域"
            )
        # 生成掩码和插值结果
        # xi_masked, yi_masked, zi_masked, zi_masked_contour = self._generate_masks()
        #
        # # Flatten the grid into 2D coordinates
        # points_2d = np.column_stack((self.xi.flatten(), self.yi.flatten()))
        # z_values = self.zs.flatten()

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        xi, yi = np.meshgrid(x, y)

        # Create a bump function (simulates a mound or hill)
        zi = np.exp(-((xi - 5) ** 2 + (yi - 5) ** 2) / 8) * 2  # smooth hill, max height ~2m

        # Optional: simulate green slope
        # zi += 0.1 * xi  # add gentle slope in x-direction
        points_2d = np.column_stack((xi.flatten(), yi.flatten()))
        z_values = zi.flatten()
        # Delaunay triangulation in 2D
        tri = Delaunay(points_2d)

        # Combine x, y, z into 3D vertices
        vertices_3d = np.column_stack((points_2d, z_values))

        # Construct mesh
        mesh = trimesh.Trimesh(vertices=vertices_3d, faces=tri.simplices)

        # Apply transformations to orient the mesh correctly (upward facing in z direction)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle=np.radians(-90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        ))
        # Export to file
        mesh.export(export_path)
        print(f"✅ 3D model saved: {export_path}")