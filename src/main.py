import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

def compute_scale(est_positions, gt_positions):
    num_points = min(len(est_positions), len(gt_positions))
    est_distances = np.linalg.norm(np.diff(est_positions[:num_points], axis=0), axis=1)
    gt_distances = np.linalg.norm(np.diff(gt_positions[:num_points], axis=0), axis=1)
    scale = np.sum(gt_distances) / np.sum(est_distances)
    return scale

def compute_rmse(est_positions, gt_positions):
    errors = []

    min_length = min(len(est_positions), len(gt_positions))

    for i in range(min_length):
        error = np.linalg.norm(est_positions[i] - gt_positions[i])
        errors.append(error)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return rmse

def plot_trajectories(est_positions, gt_positions):
    rr.init("visual_odometry", spawn=True)
    rr.log("estimated_trajectory", rr.Points3D(est_positions, colors=(255, 0, 0), radii=0.02))
    rr.log("ground_truth_trajectory", rr.Points3D(gt_positions, colors=(0, 255, 0), radii=0.02))

def compute_scale(T_est, T_gt):
    est_translation = np.linalg.norm(T_est[:3, 3])
    gt_translation = np.linalg.norm(T_gt[:3, 3])
    return gt_translation / est_translation if est_translation != 0 else 1.0

def correct_translation(T):
    T[:3, 3] = -T[:3, 3]
    return T

if __name__ == "__main__":
    v = vo.VisualOdometry()
    T = np.eye(4)
    T_gt = np.eye(4)

    for i in range(10):
        T_increment = v.run(i)
        T = T @ T_increment

        T_gt_increment = u.gt2T(np.array([v.traj[i]]))
        T_gt = T_gt @ T_gt_increment

        T = correct_translation(T)
        scale = compute_scale(T, T_gt)
        T[:3, 3] *= scale

        print(f"Frame {i}")
        print(f"T:\n{T}")
        print(f"T_gt:\n{T_gt}")
        print("======")

    