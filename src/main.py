import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

def main():
    v = vo.VisualOdometry()
    num_frames = len(v.traj)

    for idx in range(num_frames - 1):
        print(f"Esecuzione Visual Odometry per frame {idx}...")
        v.run(idx)

    estimated_poses = [u.m2T(R, t) for R, t in v.poses_camera]
    gt_poses = [u.gt2T(np.array(gt)) for gt in v.traj]

    est_positions = np.array([pose[:3, 3] for pose in estimated_poses])
    gt_positions = np.array([pose[:3, 3] for pose in gt_poses])

    rmse = compute_rmse(est_positions, gt_positions)
    print(f"RMSE tra le pose stimate scalate e la ground truth: {rmse:.4f}")

    #plot_trajectories(est_positions, gt_positions)

def compute_scale(est_positions, gt_positions):
    """
    Calcola il fattore di scala ottimale tra le pose stimate e la ground truth.
    """
    num_points = min(len(est_positions), len(gt_positions))
    est_distances = np.linalg.norm(np.diff(est_positions[:num_points], axis=0), axis=1)
    gt_distances = np.linalg.norm(np.diff(gt_positions[:num_points], axis=0), axis=1)
    scale = np.sum(gt_distances) / np.sum(est_distances)
    return scale

def compute_rmse(est_positions, gt_positions):
    """
    Calcola l'RMSE tra le posizioni stimate e quelle della ground truth.
    """
    errors = []

    min_length = min(len(est_positions), len(gt_positions))

    for i in range(min_length):
        error = np.linalg.norm(est_positions[i] - gt_positions[i])
        errors.append(error)

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    return rmse

def plot_trajectories(est_positions, gt_positions):
    """
    Plot delle traiettorie utilizzando Rerun, con etichette delle pose.
    """
    rr.init("visual_odometry", spawn=True)

    rr.log("estimated_trajectory", rr.Points3D(est_positions, colors=(255, 0, 0), radii=0.02))
    rr.log("ground_truth_trajectory", rr.Points3D(gt_positions, colors=(0, 255, 0), radii=0.02))

    for i, pos in enumerate(est_positions):
        rr.log(f"estimated_trajectory/pose_{i}", rr.Points3D([pos], colors=(255, 0, 0), labels=[f"Est {i}"]))

    for i, pos in enumerate(gt_positions):
        rr.log(f"ground_truth_trajectory/pose_{i}", rr.Points3D([pos], colors=(0, 255, 0), labels=[f"GT {i}"]))


if __name__ == "__main__":
    v = vo.VisualOdometry()
    
    T_abs = np.eye(4)
    for i in range(20):
        #print(f"T_abs:\n {T_abs}")
        T_abs = T_abs @ v.run(i)

    
    
    print(f"T_abs:\n {T_abs}")
