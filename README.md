# ros-rad4d-extract
Python 3 scripts for compiling a dataset to be used with TMVA4D. As input, the image and PCD files extracted using tmva4d-ros-extract are used.

## Getting Started
### Prerequisites
In order to run the ROS nodes, make sure that the following are installed in your Python environment: NumPy, OpenCV (CV2), Open3D. To install the packages using PIP:
```
pip3 install matplotlib numpy opencv-python open3d
```

### Installation
Clone the repo:
```
git clone https://github.com/MikaelSkog/tmva4d-compile-dataset.git
```

## Usage
### Preparation
The images and point cloud data (PCD) files extracted using tmva4d-ros-extract are taken as input in order to compile the dataset. An input directory is needed,
containing the subdirectories 'img', and 'pcd'. Both these subdirectories in turn contain subdirectories each corresponding to a given sequence from which the images
and PCD files were extracted. Note that, for a given sequence, the names of its subdirectories in 'img' and 'pcd' have to be the same. For example:
```
.
└── input_dir/
    ├── img/
    │   ├── sequence1/
    │   ├── sequence2/
    │   └── sequence3/
    └── pcd/
        ├── sequence1/
        ├── sequence2/
        └── sequence3/
```
Each sequence directory should then be populated with the images and PCD files extracted from that sequence, for the 'img' and 'pcd' respectively.

A YOLOv8 model is also necessary.

### Image Extractor
Run the script using
```
python compile_dataset.py --input_dir </path/to/input/dir/> --output_dir <path/to/output/dir/> --yolov8_model </path/to/YOLOv8/model.pt>
```
Running the program should result in a directory 'Dataset4d' being created and populated, which will be found at <path/to/output/dir/>.
