# Coco keypoint loader
## This is the repository to load coco keypoint annotations, save it in a custom json format and visualize it.

1. Install below python packages using pip:

```sh
json
numpy
tqdm
COCO
opencv-python
pytest
pytest-repeat

Install any additional dependency packages, if required.
```

(For windows)

2. For case study1, run the script from the command line as:

```sh
cocodataset.py --path path/to/coco_data or 
cocodataset.py --path path/to/coco_data --split train/val
```

3. For case study2, run the script from the command line as:
 
```sh
cocoinstancedataset.py --path path/to/coco_data or
cocoinstancedataset.py --path path/to/coco_data --split train/val --expand_bbox 0.2 --filter_on_min_area 400.0 --filter_on_min_keypoints 3 --always_crop
```
	
Alternatively, you can run both the scripts without passing any arguments from command line. Just update the default values in the script before running the scripts.
Also, run the scripts with '-h' argument to know more details about the arguments to be passed.

4. There are also scripts to visualize the dataset generated (for debug purposes).

```sh
Case study1: test_cocodataset.py
Case study2: test_cocoinstancedataset.py
```
