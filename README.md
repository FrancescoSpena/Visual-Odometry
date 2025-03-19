# Monocular Visual Odometry

Monocular Visual Odometry is a computer vision-based technique used to estimate the motion of a camera using a single input video stream. This project implements a monocular visual odometry pipeline using feature detection, matching, and pose estimation techniques (P-ICP).


## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- OpenCV
- NumPy
- Matplotlib


### Setup
Clone the repository:
```sh
$ git clone https://github.com/FrancescoSpena/Visual-Odometry/
$ cd Visual-Odometry/
```

### Usage
Run the main script with num_iter_picp (e.g. 10) iterations for P-ICP and enable the plot: 

```
$ python3 main.py --picp 10 --iter 120 --plot 1
```

Options:
- **--iter**: number of frames (total frame 120) (*Required*)
- **--picp**: number of iterations for P-ICP (*Required*)
- **--plot**: enable the plot (bool)
