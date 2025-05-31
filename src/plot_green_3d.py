from green_visualizer_3d import GreenVisualizer3D

if __name__ == "__main__":
    visualizer = GreenVisualizer3D()
    try:
        # run all the testcases
        for course_index in range(1, 4):
            for hole_index in range(1, 19):
                visualizer.plot_holes(course_index, hole_index)
                visualizer.generate_metadata(course_index, hole_index)

    finally:
        visualizer.cleanup()  # ensure resources are cleaned up
