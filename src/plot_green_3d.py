import json
from green_visualizer_3d import GreenVisualizer3D


if __name__ == "__main__":
    visualizer = GreenVisualizer3D()
    try:
        # run all the testcases
        for course_index in range(1, 2):
            holes = []
            for hole_index in range(1, 2):
                json_file = f"testcases/input/course{course_index}/{hole_index}.json"
                hole = json.load(open(json_file, "r", encoding="utf-8"))
                hole["hole_number"] = hole_index
                holes.append(hole)
            visualizer.plot_holes(
                holes, f"testcases/output/green_3d/course{course_index}"
            )
    finally:
        visualizer.cleanup()  # ensure resources are cleaned up
