import math
import numpy as np
from pygltflib import (
    GLTF2, Buffer, BufferView, Accessor, Texture, Sampler,
    PbrMetallicRoughness, Material, Primitive, Mesh, Node, Scene
)
from pygltflib.utils import Image
from green_visualizer_2d import GreenVisualizer2D
from src.utils import nearest_index, transform_coordinates, get_unique_ascending, get_duplicated_values, is_same, \
    get_indices, get_mid_point

SCALER = 1e-1
class GreenVisualizer3D(GreenVisualizer2D):
    def get_model_data(self, json_file_path):
        data = self._load_json(json_file_path)
        # Initialize data
        total_points = []
        total_indices = []
        total_texcoords = []
        side_points = []
        side_indices = []

        # Get coordinate values per axis
        # Values are sorted
        # Get min/max also
        xarr = []
        yarr = []
        zarr = []
        cxarr = []  # converted to meters
        cyarr = []
        for feature in data['features']:
            points = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'Point' and len(points) == 3:
                x, y = transform_coordinates(points[:2])
                xarr.append(points[0])
                yarr.append(points[1])
                zarr.append(points[2])
                cxarr.append(x)
                cyarr.append(y)

        xvalues, _, _ = get_unique_ascending(xarr)
        yvalues, _, _ = get_unique_ascending(yarr)
        _, zmin, _ = get_unique_ascending(zarr)
        _, cxmin, cxmax = get_unique_ascending(cxarr)
        _, cymin, cymax = get_unique_ascending(cyarr)

        xdup = get_duplicated_values(xvalues, xarr)
        ydup = get_duplicated_values(yvalues, yarr)

        # Init board
        x_count = len(xdup)
        y_count = len(ydup)
        board = [[-1] * y_count for _ in range(x_count)]

        # Fill board
        point_index = 0
        points_stored = []
        for feature in data['features']:
            points = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'Point' and len(points) == 3:
                try:
                    x_index = xdup.index(points[0])
                    y_index = ydup.index(points[1])
                    points_stored.append(points)

                    board[x_index][y_index] = point_index
                    point_index = point_index + 1
                    cx, cy = transform_coordinates(points[:2])
                    total_points = total_points + [(cx - cxmin) * SCALER, (cy - cymin) * SCALER, points[2]]
                    total_texcoords += [(cx - cxmin) / (cxmax - cxmin), 1.0 - (cy - cymin) / (cymax - cymin)]
                except:
                    pass

        # Create surface
        for i in range(x_count - 1):
            for j in range(y_count - 1):
                if board[i][j] >= 0 and board[i][j + 1] >= 0 and board[i + 1][j] >= 0 and board[i + 1][j + 1] >= 0:
                    # quad_count += 1
                    total_indices = total_indices + [
                        board[i][j],
                        board[i + 1][j],
                        board[i][j + 1],

                        board[i][j + 1],
                        board[i + 1][j],
                        board[i + 1][j + 1],
                    ]

        total_edges = []

        # process vertical edges
        for i in range(x_count):
            for j in range(y_count - 1):  # [i, j] - [i, j+1]
                if i == 0:
                    a = False
                else:
                    a = board[i - 1][j] >= 0 and board[i - 1][j + 1] >= 0 and board[i][j] >= 0 and board[i][j + 1] >= 0
                if i == x_count - 1:
                    b = False
                else:
                    b = board[i][j] >= 0 and board[i][j + 1] >= 0 and board[i + 1][j] >= 0 and board[i + 1][j + 1] >= 0

                if a ^ b:
                    total_edges += [
                        [i, j], [i, j + 1]
                    ]

        # process horizontal edges
        for i in range(x_count - 1):
            for j in range(y_count):  # [i, j] - [i+1, j]
                if j == 0:
                    a = False
                else:
                    a = board[i][j - 1] >= 0 and board[i + 1][j - 1] >= 0 and board[i][j] >= 0 and board[i + 1][j] >= 0

                if j == y_count - 1:
                    b = False
                else:
                    b = board[i][j] >= 0 and board[i + 1][j] >= 0 and board[i][j + 1] >= 0 and board[i + 1][j + 1] >= 0

                if a ^ b:
                    total_edges += [
                        [i, j], [i + 1, j]
                    ]

        # Get connected edges as a closed path
        edge_count = len(total_edges) // 2

        end_point = total_edges[0]
        prev_point = total_edges[0]
        current_point = total_edges[1]

        edge_points = [current_point]
        while True:
            found = False
            for i in range(edge_count):

                if is_same(current_point, total_edges[i * 2]) and not is_same(prev_point, total_edges[i * 2 + 1]):
                    next_point = total_edges[i * 2 + 1]
                    found = True
                    break
                if is_same(current_point, total_edges[i * 2 + 1]) and not is_same(prev_point, total_edges[i * 2]):
                    next_point = total_edges[i * 2]
                    found = True
                    break

            if not found:
                print('not found')

            prev_point = current_point
            current_point = next_point
            edge_points.append(next_point)
            if is_same(current_point, end_point):
                break

        # get board data from pnt + delta
        # returns [validity, board value]
        def check_valid(pnt, delta):
            px, py = pnt
            dx, dy = delta
            px += dx
            py += dy
            if px >= 0 and px < x_count and py >= 0 and py < y_count and board[px][py] >= 0:
                return True, board[px][py]
            return False, -1

        # returns if [grid_point, grid_point + d1, grid_point + d2, grid_point + d3] is valid quad in board
        # grid_point is already valid in board
        # returns center point of the quad as a second parameter
        # returns guessed z coords
        # (point, z) is in the extended quad
        def check(point, grid_point, d1, d2, d3):
            gx, gy = grid_point

            # center point
            index0 = board[gx][gy]
            x = points_stored[index0][0]
            y = points_stored[index0][1]
            indices = [index0]
            for d in [d1, d2, d3]:
                is_valid, index = check_valid(grid_point, d)
                if not is_valid:
                    return False, [0, 0], 0
                indices.append(index)
                x += points_stored[index][0]
                y += points_stored[index][1]
            x = x / 4
            y = y / 4

            # cross product of index0, index1, index3
            pa = np.asarray(points_stored[index0])
            pb = np.asarray(points_stored[indices[1]])
            pc = np.asarray(points_stored[indices[3]])
            va = pa - pc
            vb = pb - pc
            crss = np.cross(va, vb)  # crss = [A, B, C],  Ax + By + Cz + D = 0 is the equation of the plain
            D = -np.dot(crss, pa)

            z = -(point[0] * crss[0] + point[1] * crss[1] + D) / crss[2]

            return True, [x, y], z

        #        |
        #     2  |  1
        #    ----------
        #     3  |  4
        #        |
        # check 4 quads
        def guess_elevation(point, grid_point):
            index = 0
            result = 0
            data = [
                [[1, 0], [1, 1], [0, 1]],  # 1
                [[0, 1], [-1, 1], [-1, 0]],  # 2
                [[-1, 0], [-1, -1], [0, -1]],  # 3
                [[0, -1], [1, -1], [1, 0]]  # 4
            ]
            for i in range(4):
                is_valid, center, z = check(point, grid_point, data[i][0], data[i][1], data[i][2])
                if is_valid:
                    d = math.dist(point, center)
                    if index == 0 or d < min_dist:
                        min_dist = d
                        index = i + 1
                        result = z
            return result

        point_index = len(total_points) // 3
        point_index_store = point_index

        side_index = 0

        # Fill the remaining parts near the edge
        for feature in data['features']:
            if feature['geometry']['type'] == 'Polygon':
                points = feature['geometry']['coordinates']
                point_count = len(points)
                # print('polygon', point_count)

                for i in range(point_count):
                    next_i = 0 if i == point_count - 1 else i + 1

                    a = nearest_index(points[i], edge_points, xdup, ydup)
                    b = nearest_index(points[next_i], edge_points, xdup, ydup)

                    c = get_indices(a, b, len(edge_points))
                    if len(c) == 1:
                        # add a triangle
                        nindex = nearest_index(points[i], edge_points, xdup, ydup)
                        z = guess_elevation(points[i], edge_points[nindex])
                        cx, cy = transform_coordinates(points[i])
                        total_points += [(cx - cxmin) * SCALER, (cy - cymin) * SCALER, z]
                        total_texcoords += [(cx - cxmin) / (cxmax - cxmin), 1.0 - (cy - cymin) / (cymax - cymin)]
                        next_index = point_index_store if i + 1 == point_count else point_index + 1
                        total_indices += [point_index,
                                          board[edge_points[a][0]][edge_points[a][1]],
                                          next_index]
                        point_index += 1

                        # add side
                        side_points += [(cx - cxmin) * SCALER, (cy - cymin) * SCALER, z,
                                        (cx - cxmin) * SCALER, (cy - cymin) * SCALER, zmin]
                        next_index = 0 if i + 1 == point_count else side_index + 2
                        side_indices += [
                            side_index,  # 0
                            next_index,  # 1
                            side_index + 1,  # 2
                            next_index,  # 1
                            next_index + 1,  # 3
                            side_index + 1,  # 2
                        ]
                        side_index += 2


                    else:
                        # add quads
                        start_point = points[i]
                        for j in range(1, len(c)):  # 1 ~ N -1
                            next_point = points[next_i] if j == len(c) - 1 else get_mid_point(points[i], points[next_i],
                                                                                              j / (len(c) - 1))

                            nindex = nearest_index(start_point, edge_points, xdup, ydup)
                            z = guess_elevation(start_point, edge_points[nindex])
                            cx, cy = transform_coordinates(start_point)
                            total_points += [(cx - cxmin) * SCALER, (cy - cymin) * SCALER, z]
                            total_texcoords += [(cx - cxmin) / (cxmax - cxmin), 1.0 - (cy - cymin) / (cymax - cymin)]

                            start_point = next_point
                            next_index = point_index_store if j + 1 == len(
                                c) and i + 1 == point_count else point_index + 1

                            total_indices += [
                                point_index,  # 0
                                next_index,  # 1
                                board[edge_points[c[j - 1]][0]][edge_points[c[j - 1]][1]],  # 2
                                next_index,  # 1
                                board[edge_points[c[j]][0]][edge_points[c[j]][1]],  # 3
                                board[edge_points[c[j - 1]][0]][edge_points[c[j - 1]][1]],  # 2
                            ]

                            # add side
                            side_points += [(cx - cxmin) * SCALER, (cy - cymin) * SCALER, z,
                                            (cx - cxmin) * SCALER, (cy - cymin) * SCALER, zmin]
                            next_index = 0 if j + 1 == len(c) and i + 1 == point_count else side_index + 2
                            side_indices += [
                                side_index,  # 0
                                next_index,  # 1
                                side_index + 1,  # 2
                                next_index,  # 1
                                next_index + 1,  # 3
                                side_index + 1,  # 2
                            ]
                            side_index += 2
                            point_index += 1

        for i in range(len(total_indices) // 3):
            indices = total_indices[i * 3: i * 3 + 3]
            points = []
            for k in range(3):
                index = indices[k]
                point = np.asarray(total_points[index * 3: index * 3 + 2])
                points.append(point)
            crss = np.cross(points[1] - points[0], points[2] - points[0])
            if crss < 0:
                total_indices[i * 3 + 1], total_indices[i * 3 + 2] = indices[2], indices[1]

        return total_points, total_indices, total_texcoords, side_points, side_indices

    def plot_holes(self, course_index, hole_index):
        gltf = GLTF2()

        output_name = f"testcases/output/green_3d/course{course_index}/{hole_index}.glb"

        # Add texture
        with open(f"testcases/output/green_2d/course{course_index}/{hole_index}.png", 'rb') as f:
            texture_data = f.read()

        # Get mesh data from json file
        vertices, indices, texcoords, vertices2, indices2 = self.get_model_data(
            f"testcases/input/course{course_index}/{hole_index}.json"
        )

        # Add vertices data
        vertices_data = np.array(vertices, dtype=np.float32).tobytes()
        vertices2_data = np.array(vertices2, dtype=np.float32).tobytes()
        # Define vertex indices for two triangles that make up the quad
        indices_data = np.array(indices, dtype=np.int16).tobytes()
        indices2_data = np.array(indices2, dtype=np.int16).tobytes()
        # Add texcoords
        texcoords_data = np.array(texcoords, dtype=np.float32).tobytes()

        # Create a buffer
        buffer_data = vertices_data + indices_data + texcoords_data + vertices2_data + indices2_data + texture_data
        gltf.buffers.append(Buffer(byteLength=len(buffer_data)))

        # Create a buffer view
        vertex_view = BufferView(buffer=0, byteOffset=0, byteLength=len(vertices_data), target=34962)  # Vertex buffer
        index_view = BufferView(buffer=0, byteOffset=len(vertices_data), byteLength=len(indices_data),
                                target=34963)  # Index buffer
        texcoord_view = BufferView(buffer=0, byteOffset=index_view.byteOffset + len(indices_data),
                                   byteLength=len(texcoords_data), target=34962)
        vertex2_view = BufferView(buffer=0, byteOffset=texcoord_view.byteOffset + len(texcoords_data),
                                  byteLength=len(vertices2_data), target=34962)
        index2_view = BufferView(buffer=0, byteOffset=vertex2_view.byteOffset + len(vertices2_data),
                                 byteLength=len(indices2_data), target=34963)  # Index buffer
        image_view = BufferView(buffer=0, byteOffset=index2_view.byteOffset + len(indices2_data),
                                byteLength=len(texture_data))  # ✅ No target

        gltf.bufferViews.extend([vertex_view, index_view, texcoord_view, vertex2_view, index2_view, image_view])

        # === Accessors ===
        # Helper to compute bounds
        def compute_bounds(arr):
            arr = np.array(arr).reshape(-1, 3)
            return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()

        min1, max1 = compute_bounds(vertices)
        min2, max2 = compute_bounds(vertices2)

        # Now create accessors with bounds
        gltf.accessors.append(Accessor(
            bufferView=0, byteOffset=0, componentType=5126, count=len(vertices) // 3, type="VEC3",
            min=min1, max=max1
        ))  # POSITION accessor 0

        gltf.accessors.append(Accessor(
            bufferView=1, byteOffset=0, componentType=5123, count=len(indices), type="SCALAR"
        ))  # INDICES accessor 1

        gltf.accessors.append(Accessor(
            bufferView=2, byteOffset=0, componentType=5126, count=len(texcoords) // 2, type="VEC2"
        ))  # TEXCOORD accessor 2

        gltf.accessors.append(Accessor(
            bufferView=3, byteOffset=0, componentType=5126, count=len(vertices2) // 3, type="VEC3",
            min=min2, max=max2
        ))  # POSITION 2 accessor 3

        gltf.accessors.append(Accessor(
            bufferView=4, byteOffset=0, componentType=5123, count=len(indices2), type="SCALAR"
        ))  # INDICES 2 accessor 4

        # === Image, Sampler, Texture ===
        gltf.images.append(Image(bufferView=5, mimeType='image/png'))
        gltf.samplers.append(Sampler(minFilter=9729, magFilter=9729, wrapS=10497, wrapT=10497))
        gltf.textures.append(Texture(sampler=0, source=0))

        # === Materials ===
        pbr = PbrMetallicRoughness(baseColorTexture={"index": 0, "texCoord": 0}, metallicFactor=0.1)
        material = Material(pbrMetallicRoughness=pbr, doubleSided=True)
        material2 = Material(doubleSided=True)
        gltf.materials.extend([material, material2])

        # === Mesh Primitives ===
        primitive1 = Primitive(attributes={"POSITION": 0, "TEXCOORD_0": 2}, indices=1, material=0)
        primitive2 = Primitive(attributes={"POSITION": 3}, indices=4, material=1)
        gltf.meshes.append(Mesh(primitives=[primitive1, primitive2]))

        # === Node and Scene ===
        gltf.nodes.append(Node(mesh=0))
        gltf.scenes.append(Scene(nodes=[0]))
        gltf.scene = 0

        # === Finalize GLB ===
        gltf.set_binary_blob(buffer_data)
        gltf.save(output_name)