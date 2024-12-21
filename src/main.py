import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

def load_groundtruth_poses(traj_path):
    gt_positions = []
    with open(traj_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 7:
                x, y, z = map(float, values[4:7])
                gt_positions.append(np.array([x, y, z]))
    return gt_positions

def compute_relative_scale(est_translation, gt_translation):
    est_norm = np.linalg.norm(est_translation)
    gt_norm = np.linalg.norm(gt_translation)
    return gt_norm / est_norm if est_norm != 0 else 1.0

def plot_rerun(est_positions,gt_traj_positions):
    rr.init("visual_odometry", spawn=True)
    rr.log("estimated_trajectory", rr.Points3D(est_positions, colors=(255, 0, 0), radii=0.02))
    rr.log("ground_truth_trajectory", rr.Points3D(gt_traj_positions, colors=(0, 255, 0), radii=0.02))

def compute_rotation(est_positions, gt_positions):
    est_centered = est_positions - np.mean(est_positions, axis=0)
    gt_centered = gt_positions - np.mean(gt_positions, axis=0)

    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R_optimal = Vt.T @ U.T

    if np.linalg.det(R_optimal) < 0:
        Vt[-1, :] *= -1
        R_optimal = Vt.T @ U.T

    return R_optimal


if __name__ == "__main__":
    v = vo.VisualOdometry()
    gt_positions = load_groundtruth_poses('../data/trajectoy.dat')
    iter = len(gt_positions)

    T = np.eye(4)
    T_gt = np.eye(4)

    est_positions = [T[:3, 3].copy()]
    gt_traj_positions = [gt_positions[0]]

    for i in range(1, iter):
        T_increment = v.run(i - 1)
        T = T @ T_increment

        gt_translation_rel = gt_positions[i] - gt_positions[i - 1]
        est_translation_rel = T[:3, 3] - est_positions[-1]

        scale = compute_relative_scale(est_translation_rel, gt_translation_rel)
        T[:3, 3] = est_positions[-1] + scale * est_translation_rel

        est_positions.append(T[:3, 3].copy())
        gt_traj_positions.append(gt_positions[i])

    est_positions = np.array(est_positions)
    gt_traj_positions = np.array(gt_traj_positions)

    # Calcolare la rotazione e applicarla
    R_correct = compute_rotation(est_positions, gt_traj_positions)
    est_positions_rotated = (R_correct @ est_positions.T).T

    # Plottare i punti rotati
    plot_rerun(est_positions_rotated, gt_traj_positions)
