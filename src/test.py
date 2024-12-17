import rerun as rr
import numpy as np


def load_trajectory_data(file_path):
    odometry_poses = []
    gt_poses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip(): 
                data = line.split()

                odo_x, odo_y, odo_z = map(float, data[1:4])
                gt_x, gt_y, gt_z = map(float, data[4:7])
                odometry_poses.append([odo_x, odo_y, odo_z])
                gt_poses.append([gt_x, gt_y, gt_z])
    return np.array(odometry_poses), np.array(gt_poses)


trajectory_path = "../data/trajectoy.dat"  
odometry_poses, gt_poses = load_trajectory_data(trajectory_path)

rr.init("trajectory_visualization", spawn=True)
rr.log("odometry", rr.LineStrips3D([odometry_poses], colors=[255, 0, 0]))
rr.log("ground_truth", rr.LineStrips3D([gt_poses], colors=[0, 255, 0]))
