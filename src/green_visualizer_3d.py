import numpy as np
import os
import trimesh

from green_visualizer_2d import GreenVisualizer2D

class GreenVisualizer3D(GreenVisualizer2D):
    def _plot(self):
        xi_masked, yi_masked, zi_masked, _ = self._generate_masks()

        # Step 1: Create vertices and faces
        rows, cols = xi_masked.shape
        z_exaggeration = 10  # Adjust this factor as needed
        zi_masked = zi_masked * z_exaggeration
        vertices = np.column_stack([
            xi_masked.filled(0).ravel(),
            yi_masked.filled(0).ravel(),
            zi_masked.filled(0).ravel()
        ])
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                a = i * cols + j
                b = a + 1
                c = a + cols
                d = c + 1
                if not (np.ma.is_masked(zi_masked[i, j]) or
                        np.ma.is_masked(zi_masked[i + 1, j]) or
                        np.ma.is_masked(zi_masked[i, j + 1])):
                    faces.append([a, b, c])
                if not (np.ma.is_masked(zi_masked[i + 1, j + 1]) or
                        np.ma.is_masked(zi_masked[i + 1, j]) or
                        np.ma.is_masked(zi_masked[i, j + 1])):
                    faces.append([b, d, c])
        faces = np.array(faces)

        # Create the mesh with visuals
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            angle=np.radians(-90),
            direction=[1, 0, 0],
            point=[0, 0, 0]
        ))
        output_png_path = f"{self.output_path}/{self.hole_number}.glb"
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        mesh.export(output_png_path)
