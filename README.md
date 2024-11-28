
# 1. Vehicle Counting Repository Setup Guide

This guide provides step-by-step instructions to set up and run the vehicle counting application on three different platforms: Google Colab, Jupyter Notebooks, and via Bash/Linux commands.

## 1.1. Google Colab

If you prefer working with Colab notebooks, click on the icon below to open the notebook in Google Colab:

[![Open In Colab](pictures/colab_icon.png)](https://colab.research.google.com/drive/1ZCVPWJOPqZMeieotSRxd9JSj-YZUI12j?usp=sharing)

**Note:** Set the runtime type to **GPU (T4)** in Colab for optimal performance.

### Setting the Runtime to GPU (T4)

1. Open the notebook, then go to **Runtime** > **Change runtime type**.
2. In the pop-up window, set **Hardware accelerator** to **GPU**.
3. If available, select **T4** as the GPU type.

Follow the notebook instructions. Feel free to explore and customize the code to suit your needs. The notebook is designed to be user-friendly and flexible!

## 1.2. Jupyter Notebooks

Follow these steps to set up and run the application in Jupyter Notebooks:

### Step 1: Create a Virtual Environment (Bash/Anaconda Prompt)

Open a Bash or Anaconda Prompt and run the following commands to create and activate a virtual environment named `vehicle_counter`:

```bash
conda create --name vehicle_counter python=3.8
conda activate vehicle_counter
```

This step assumes you have already installed Anaconda on your computer.

> **Note:** Skip the above step if you are not working in a virtual environment.

### Step 2: Clone the Repository

Download/clone the repository and set `vehicle_counting_CV` as your working directory if you haven't already:

```bash
git clone https://github.com/hamzaelouiaazzani/vehicle_counting_CV.git
cd vehicle_counting_CV
```

### Step 3: Install Dependencies

Run the following commands to upgrade pip, setuptools, and wheel, and install the repository dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

> **Note:** After running this step, comment out the installation commands to avoid reinstalling the packages unnecessarily.

### Step 4: Verify PyTorch Installation

Run the following Python code to confirm that PyTorch and CUDA are installed correctly:

```python
import torch
print(f"CUDA availability: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name})")
```

### Step 5: Run Vehicle Counting

**Import the necessary modules:**

```python
from counting.run_count import run
from counting.count import args
```

**Customize the arguments to configure your vehicle counting system:**

```python
args.source = "kech.mp4"
args.name = "video_results_folder"
args.counting_approach = "tracking_with_line_crossing"
args.tracking_method = "ocsort"
args.save = True
args.verbose = True
args.line_vicinity = 1.5
args.line_point11 = (0.0, 0.25)
args.line_point12 = (1.0, 0.75)
```

**Run the counting process:**

```python
counter_yolo, profilers, _ = run(args)
```

Results will be saved in the `runs/count/video_results_folder` directory if `args.save` is set to `True`.

**Extract video attributes:**

```python
print(counter_yolo.video_attributes)
```

**Display total vehicle counts:**

```python
print(f"Total vehicles counted: {counter_yolo.counter}")
```

**Display counts per vehicle type:**

```python
def tensor_to_dict(count_per_class):
    vehicle_types = ["bicycle", "car", "motorcycle", "bus", "truck"]
    indices = [1, 2, 3, 5, 7]
    return {vehicle: int(count_per_class[idx].item()) for vehicle, idx in zip(vehicle_types, indices)}

print(f"Vehicles per type: {tensor_to_dict(counter_yolo.count_per_class)}")
```

## 1.3. Bash/Linux Commands

### Step 1: Create a Virtual Environment

Open a terminal and run the following commands:

```bash
python -m venv vehicle_counter
source vehicle_counter/bin/activate
```

### Step 2: Install Dependencies

Run the following commands:

```bash
pip install --upgrade pip setuptools wheel
git clone https://github.com/hamzaelouiaazzani/vehicle_counting_CV.git
cd vehicle_counting_CV
pip install -e .
```

### Step 3: Verify PyTorch Installation

Run the following command:

```bash
python -c "import torch; print(f'Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})')"
```

### Step 4: Run Vehicle Counting

Edit the `args` configuration in the `demo.py` file and run:

```bash
python3 counting/demo.py
```

Results will be saved in the `runs/count` folder if `args.save` is set to `True`.

---

# 2. Configuring Arguments for Optimal Performance

This repository offers configurable options to tailor the counting process. Execute the following command to explore the arguments:

```python
from counting.count import args
print(args.__doc__)
```

Refer to the documentation for detailed descriptions of each argument.
