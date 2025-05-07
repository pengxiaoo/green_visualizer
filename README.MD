## Project Overview

This project is a tool to visualize the 3d coordinates of a golf green and convert them to a 2d map or 3d model. the 2d map or 3d model will be
a contour map with color gradient, contour lines, and gradient arrows.

the input are json files, each file contains the 3d coordinates of a golf green. example can be found
in `testcases/input`

### Green 2d
the output are png files, each file is a 2d map of a golf green, for example:

<img src="images/example_contour_map_1.png" width="300">

<img src="images/example_contour_map_2.png" width="300">

### Green 3d
the output are 3d models, and we need both glb and usdz format.

there is an example of 3d models in [model-examples](./model-examples)

## Key requirements

- the 2d map or 3d model should be accurately represent the 3d coordinates of the golf green
- the edge should be smooth
- the gradient arrows should be evenly distributed, not overlapped with each other, not crossing the edge, and the density should be adjustable

## Set up Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
