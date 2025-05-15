import json
from green_visualizer_2d import GreenVisualizer2D


if __name__ == "__main__":
    visualizer = GreenVisualizer2D()
    try:
        # run all the testcases
        for course_index in range(1, 4):
            holes = []
            for hole_index in range(1, 19):
                json_file = f"testcases/input/course{course_index}/{hole_index}.json"
                hole = json.load(open(json_file, "r", encoding="utf-8"))
                hole["hole_number"] = hole_index
                holes.append(hole)
            visualizer.plot_holes(
                holes, f"testcases/output/green_2d/course{course_index}"
            )
    finally:
        visualizer.cleanup()  # 确保最后清理资源
