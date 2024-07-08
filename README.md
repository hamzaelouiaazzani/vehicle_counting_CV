# 1. Vehicle Counting Repository Setup Guide
This notebook provides step-by-step instructions to set up and run the vehicle counting application on three different platforms: Google Colab, Jupyter Notebooks, and via Bash/Linux commands.
## 1.1. Google Colab
Follow these steps to set up and run the application in Google Colab.

### Step 1: Install NumPy
Run the following cell to install NumPy version 1.24.4.


```python
!pip install numpy==1.24.4
```

If you see a message like the following, click **RESTART SESSION** and then re-run the previous and upcoming cells.

![Restart Session 1](pictures/restart_session1.png)
![Restart Session 2](pictures/restart_session2.png)


### Step 2: Verify NumPy Installation
Run the following cell to ensure NumPy version 1.24.4 is installed.


```python
import numpy as np
print("NumPy Version:", np.__version__)
```

### Step 3: Clone Repository and Install Dependencies
Run the following cell to upgrade pip, setuptools, and wheel, clone the repository and install dependencies and finally .


```python
!pip install --upgrade pip setuptools wheel
!git clone https://github.com/hamzaelouiaazzani/vehicle_counting_CV.git  # clone repo
!pip install -e .
import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

### Step 4: Run counting: 

Import counting files:


```python
from counting.run_count import run
from counting.count import args
```

To print the documentation about the arguments:


```python
print(args.__doc__)
```

Change the arguments as you want:


```python
args.source = "kech.mp4"
args.name = "kech"
args.verbose = True
args.save_csv_count = True
```

Run the counting process:


```python
counter_yolo , profilers , results  = run(args)
```

if you prompt args.save to **True** please check the results saved in folder **\runs\count** that is going to be created.

- counter_yolo object is the object that stores everything about the detector, the tracking and the counting used algorithms.
- profilers: contains the time taken by each phase of the pipeline for the given video: **Pre-processing----> Detection----> Post-processing----> Tracking----> Counting**.
- results: contains the results of detection and tracking of diffrents objects of the video.

Extract informations:


```python
# video attribues:
counter_yolo.video_attributes
```


```python
# Counting Results
print(f"The number of vehicles counted by the algorithm is: {counter_yolo.counter}")
print(f"The number of vehicles per type counted by the algorithm is: {counter_yolo.count_per_class}")
```

## 1.2. Jupyter Notebooks
Follow these steps to set up and run the application in Jupyter Notebooks.

### Step 1: Create Virtual Environment (Bash/Anaconda Prompt)
Open a Bash or Anaconda Prompt and run the following commands to create and activate a virtual environment named `vehicle_counter`:
```bash
conda create --name vehicle_counter python=3.8
conda activate vehicle_counter
```

> **Note:** You can neglect the above two instructions if you don't want working in a virtual envirenement.

### Step 2: Upgrade pip and Install Dependencies
Download/clone the repository and run the following cell to upgrade pip, setuptools, and wheel, and install the repository dependencies.


```python
!pip install --upgrade pip setuptools wheel
!pip install -e .
```

> **Note:** Once you run this cell, comment it out and do not run it again because the packages are already installed in the environment.

### Step 3: Verify Torch Installation
Run the following cell to verify the installation of PyTorch and check if CUDA is available.


```python
import torch
if torch.cuda.is_available():
    torch.cuda.get_device_properties(0).name
torch.cuda.is_available()
```

### Step 4: Clear Output and Confirm Setup
Run the following cell to clear the output and confirm the setup.


```python
from IPython.display import Image, clear_output  # to display images

clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

### Step 5: Run counting

Import counting files:


```python
from counting.run_count import run
from counting.count import args
```

To print the documentation about the arguments:


```python
print(args.__doc__)
```

Change the arguments as you want:


```python
args.source = "kech.mp4"
args.name = "kech"
args.verbose = True
args.save_csv_count = True
```

Run the counting process:


```python
# Run the counting algorithm
counter_yolo , profilers , results  = run(args)
```

if you prompt args.save to **True** please check the results saved in folder **\runs\count** that is going to be created.

- counter_yolo: This object stores all information about the detector, tracking, and counting algorithms used.
- profilers: This object contains the time taken by each phase of the pipeline for the given video: **Pre-processing----> Detection----> Post-processing----> Tracking----> Counting**.
- results: This object contains the results of detection and tracking for different objects in the video.

Extract informations:


```python
# video attribues:
counter_yolo.video_attributes
```


```python
# Counting Results
print(f"The number of vehicles counted by the algorithm is: {counter_yolo.counter}")
print(f"The number of vehicles per type counted by the algorithm is: {counter_yolo.count_per_class}")
```

## 1.3. Bash/Linux Commands
Follow these steps to set up and run the application via Bash/Linux commands.

### Step 1: Create Virtual Environment
Open a terminal and run the following commands to create and activate a virtual environment named `vehicle_counter`:
```bash
python -m venv vehicle_counter
source vehicle_counter/bin/activate
```

### Step 2: Upgrade pip and Install Dependencies
Run the following commands to upgrade pip, setuptools, and wheel, and install the repository dependencies:
```bash
pip install --upgrade pip setuptools wheel
git clone https://github.com/hamzaelouiaazzani/vehicle_counting_CV.git  # clone repo
pip install -e .
```

### Step 3: Verify Torch Installation
Run the following commands to verify the installation of PyTorch and check if CUDA is available:
```bash
python -c "import torch; print(f'Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})')"
```

### Step 4: Run counting: 
Kindly before running the following counting script be sure you prompt your chosen **args** configuration within the demo.py file (remember you can edit your configuration as you want):
```bash
python3 counting/demo.py
```
If you prompt args.save to **True** please check the results saved in folder **\runs\count** that is going to be created.

# 2. Configuring Your Arguments for Optimal Counting Performance

This repository offers a wide range of configurable features to tailor the counting process to your specific use-case. You can customize the algorithm with various arguments, select your preferred detector, tracker, or counting approach, and flexibly set the counting lines using percentages. Additionally, you can apply spatial masks to focus on specific areas of your video frames and more.

To explore and understand these configurable arguments in detail, please execute the following commnd:



```python
from counting.count import args
print(args.__doc__)
```

    
        This class contains configuration parameters for the vehicle counting system using the YOLO model and various tracking approaches.
    
        Attributes:
            source (str): Filename of the video to perform counting on.
                          Need to be set.
            name (str): Name of the folder for the current experiment results.
                        Need to be set.
            yolo_model (Path): Path to the YOLO model file.
                               Default is 'yolov8n.pt'.
            reid_model (Path): Path to the re-identification model file used if the tracker employs appearance description of objects.
                               Examples include 'osnet_x0_25_market1501.pt', 'mobilenetv2_x1_4_msmt17.engine', etc.
            tracking_method (str): Method used for tracking. Options include 'bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', and 'hybridsort'.
            imgsz (list): Input size of the frames.
                          Default is [640].
            conf (float): Confidence threshold for detection.
                          Default is 0.6.
            iou (float): Intersection over Union (IoU) threshold.
                         Default is 0.7.
            device (str): Device used for running the model (GPU by default).
                          Default is ''.
            show (bool): Whether to display the video scene. Not supported in Google Colab.
                         Default is False.
            save (bool): Whether to save the videos illustrating the tracking results.
                         Default is True.
            classes (list): List of class indices to detect.
                            Default is [1, 2, 3, 5, 7] (vehicles).
            project (str): Folder to save the tracking results.
                           Default is 'runs/count'.
            exist_ok (bool): Whether to overwrite existing results.
                             Default is True.
            half (bool): Whether to use half-precision (16-bit floating-point format) to reduce memory consumption.
                         Default is False.
            vid_stride (int): Frame stride, e.g., process all frames with stride=1 or process every other frame with stride=2.
                              Default is 1.
            show_labels (bool): Whether to display labels (e.g., car, truck, bus) in the saved video results.
                                Default is True.
            show_conf (bool): Whether to display confidence scores of detections.
                              Default is False.
            save_txt (bool): Whether to save results in a text file format.
                             Default is False.
            save_id_crops (bool): Whether to save tracking results for each object in frames.
                                  Default is True.
            save_mot (bool): Whether to save tracking results in a report file.
                             Default is True.
            line_width (int): Line width of the bounding boxes.
                              Default is None.
            per_class (bool): Whether to count per class.
                              Default is True.
            verbose (bool): Whether to enable verbose logging.
                            Default is False.
            counting_approach (str): Approach for counting vehicles. Options include 'detection_only', 'tracking_without_line', 'tracking_with_line', 'tracking_with_two_lines'.
                                     Default is 'tracking_with_two_lines'.
            line_point11 (tuple): Coordinates of the first point of the first line. Values between 0 and 1 indicate percentages.
                                  For example, (0.4, 0.0) means 40% of the frame width (pixel 0.4 * image width) and 0% of the frame height (pixel 0).
                                  When masking the video frames with included_box, it becomes 0.4 * new width after mask.
            line_point12 (tuple): Coordinates of the second point of the first line. Values between 0 and 1 indicate percentages.
                                  For example, (0.3, 1.0) means 30% of the frame width (pixel 0.3 * image width) and 100% of the frame height (pixel image height).
            line_vicinity (float): Vicinity of the line for counting. This argument is used in the 'detection_only' or 'tracking_with_line' counting approaches and ignored otherwise ('tracking_without_line' or 'tracking_with_two_lines').
                                   Default is 0.1.
            line_point21 (tuple): Coordinates of the first point of the second line. Values between 0 and 1 indicate percentages.
                                  For example, (0.6, 0.0) means 60% of the frame width (pixel 0.6 * image width) and 0% of the frame height (pixel 0).
                                  This argument is considered only in the 'tracking_with_two_lines' counting approach and ignored otherwise.
            line_point22 (tuple): Coordinates of the second point of the second line. Values between 0 and 1 indicate percentages.
                                  For example, (0.7, 1.0) means 70% of the frame width (pixel 0.7 * image width) and 100% of the frame height (pixel image height).
                                  This argument is considered only in the 'tracking_with_two_lines' counting approach and ignored otherwise.
            use_mask (bool): Whether to use a mask for preprocessing. If set to False, 'visualize_masked_frames' and 'included_box' arguments will be ignored.
                             If set to True, the percentages for 'line_point11', 'line_point12', 'line_point21', and 'line_point22' will be transformed to pixel values with respect to the included_box.
                             Default is False.
            visualize_masked_frames (bool): Whether to visualize masked frames.
                                            Default is True.
            included_box (list): Box coordinates for masking, specified as percentages between -1 and 1. For example, [0.1, 0.2, -0.2, -0.1] indicates:
                                 - The first two values (0.1, 0.2) represent the TOP-LEFT point of the included rectangle when using a mask for frames. 
                                   This point is 10% of the width and 20% of the height.
                                 - The last two values (-0.2, -0.1) represent the BOTTOM-RIGHT point of the included rectangle after masking. 
                                   This point is 80% of the width and 90% of the height.
        
    

## 2.1. Counting Approach Configuration

The first important argument to configure is `counting_approach`, which determines the method used to count vehicles in the road traffic video scene. This framework offers four counting approaches:

1. **detection_only**
2. **tracking_without_line**
3. **tracking_with_line**
4. **tracking_with_two_lines**

### 2.1.1. detection_only
This approach uses only a detector without a tracker. Each vehicle that crosses a predefined line - which can be set easily by specifying two points using the arguments `args.line_point11` and `args.line_point12` with percentages as described in the documentation above - is counted as a new vehicle. A vehicle is considered to have crossed the line if the distance between the line and the center point of the detected bounding box is less than the threshold `line_vicinity`, which can be manually set via the argument `args.line_vicinity`.

**Pros:**
- More efficient as it doesn't use a tracker, reducing processing time.

**Cons:**
- Vehicles may be over-counted if they appear in multiple consecutive frames.
- This approach does not support the speed estimation algorithm.
- Setting the reference line is not generic and depends on the specific traffic flow. You will need to adjust the line placement for each traffic scene based on the road traffic flow.

To mitigate the inaccuracy of this approach, you can reduce the `line_vicinity` threshold to nearly zero. However, a very small threshold may result in under-counting as some vehicles might not be counted. Alternatively, you can adjust the `vid_stride` argument to 2, 3, or 4, depending on the frame rate of your video. Generally:
  - Increasing `line_vicinity` leads to more over-counting.
  - Decreasing `line_vicinity` leads to more under-counting.
  - Higher `vid_stride` values (e.g., 2, 3, 4) result in more information loss, causing high-speed vehicles to be under-counted and low-speed vehicles to be over-counted.

### 2.1.2. tracking_without_line
This approach uses both a detector and a tracker to count vehicles, without relying on any reference lines. Instead, it is based on the unique IDs assigned by the tracker to each vehicle. The vehicle count in this approach corresponds to the number of unique IDs generated by the tracking algorithm. The accuracy of the count depends on the performance of the tracker in consistently assigning unique IDs across different frames.

While high-performing trackers generally yield accurate results, even minor inaccuracies can lead to ID switches in complex scenes with overlapping, congestion, or fast-moving objects, resulting in significant counting errors.

To mitigate these issues:
- Increase the `vid_stride` argument to 2, 3, 4, or even 5, depending on your video scene.
- Consider using more advanced trackers via the `tracking_method` argument.
- Set `use_mask` to True and focus on specific areas of the video where counting is more reliable by defining the spatial box coordinates (of your focus) using the `included_box` argument to focus detection and tracking as described in the documentation.

**Pros:**
- More accurate than the "detection_only" approach.

**Cons:**
- Susceptible to ID switch issues in complex scenes, leading to over-counting.
- Speed estimation is not supported.

### 2.1.3. tracking_with_line
This approach combines elements from both the `detection_only` and `tracking_without_line` methods. It utilizes a detector, a tracker, and a reference line to count vehicles. This method aims to mitigate the shortcomings of the `detection_only` approach (such as over-counting and under-counting) and the `tracking_without_line` approach (such as ID switch issues).

In the `tracking_with_line` approach, counting is performed only when vehicles cross a predefined line. By setting a larger `line_vicinity`, under-counting can be avoided. Tracking within the vicinity of the line helps to prevent over-counting.

**Pros:**
- Very accurate, performing well even in complex road traffic environments.

**Cons:**
- Setting the reference line is not generic and depends on the specific traffic flow. For each different traffic scene, you will need to adjust the line placement based on the road traffic flow.
- Speed estimation is not supported.

**How to Set Up:**
- Define the reference line using `args.line_point11` and `args.line_point12`.
- Adjust the `line_vicinity` to ensure accurate counting.

This approach effectively mitigates the issues of over-counting and under-counting seen in the `detection_only` method and the ID switch phenomena observed in the `tracking_without_line` method.

### 2.1.4. tracking_with_two_lines
This approach extends the `tracking_with_line` method. Instead of tracking vehicles in the vicinity of a single line, it tracks vehicles between two predefined lines. You can set these lines using the arguments `line_point11`, `line_point12`, `line_point21`, and `line_point22`.

**Pros:**
- Includes all the advantages of the `tracking_with_line` approach.
- Supports the speed estimation algorithm.

**Cons:**
- Setting the reference lines is not generic and depends on the specific traffic flow. For each different traffic scene, you will need to adjust the line placements based on the road traffic flow.

**How to Set Up:**
- Define the two reference lines using `args.line_point11`, `args.line_point12`, `args.line_point21`, and `args.line_point22`.

This approach combines the strengths of the `tracking_with_line` method while enabling speed estimation, making it suitable for complex road traffic environments.



```python

```
