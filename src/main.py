from VisualOdometry import VisualOdometry
import numpy as np
import utils as u
import rerun as rr
from scipy.spatial.transform import Rotation

def check_poses(vo, iter):
    print("Check pose...")
    scale_ratios = []
    gt = vo.gt

    for i in range(iter):
        T_curr = vo.run(i)
        T_gt_curr = u.gt2T(gt[i])

        if i > 0:
            rel_T = np.linalg.inv(T_prev) @ T_curr
            rel_GT = np.linalg.inv(T_gt_prev) @ T_gt_curr

            norm_rel_T = np.linalg.norm(rel_T[:3, 3])
            norm_rel_GT = np.linalg.norm(rel_GT[:3, 3])

            if norm_rel_GT != 0:
                scale_ratios.append(norm_rel_T / norm_rel_GT)
          
        T_prev = T_curr
        T_gt_prev = T_gt_curr

    scale_ratio = np.mean(scale_ratios)
    return scale_ratio

def check_map(vo, iter, scale_ratio):
    print("Test map:")
    gt = vo.gt  
    estimated_poses = []  
    gt_poses = []         

    T_abs = np.eye(4)

    for i in range(iter):
        T_rel = vo.run(i)
        T_abs = T_abs @ T_rel
        T_gt_curr = u.gt2T(gt[i])

        scaled_translation = T_abs[:3, 3] * scale_ratio

        estimated_poses.append(scaled_translation)
        gt_poses.append(T_gt_curr[:3, 3])

    estimated_poses = np.array(estimated_poses)
    gt_poses = np.array(gt_poses)

    errors = np.linalg.norm(estimated_poses - gt_poses, axis=1)
    return np.sqrt(np.mean(errors ** 2))


def visualize_trajectories(vo, iter):
    rr.init("visualize_trajectories", spawn=True)
    
    gt = vo.gt  
    estimated_poses = []  
    gt_poses = []

    T_abs = np.eye(4)

    for i in range(iter):
        T_rel = vo.run(i)
        T_abs = T_abs @ T_rel 

        estimated_poses.append(T_abs[:3, 3])

        T_gt_curr = u.gt2T(gt[i])
        gt_poses.append(T_gt_curr[:3, 3])

    # Convert to numpy arrays for visualization
    estimated_poses = np.array(estimated_poses)
    gt_poses = np.array(gt_poses)

    # Log the estimated trajectory
    rr.log("estimated_trajectory", rr.Points3D(estimated_poses, colors=[255, 0, 0]))

    # Log the ground truth trajectory
    rr.log("ground_truth_trajectory", rr.Points3D(gt_poses, colors=[0, 255, 0]))

if __name__ == '__main__':
    iter=100
    optim=50
    
    # vo = VisualOdometry(optim=optim)
    # ratio = check_poses(vo,iter)
    # print(f"ratio = {ratio}")

    # #I need to reconstruct the obj to reset all the internal parameters
    vo = VisualOdometry(optim=optim)
    rmse = check_map(vo,iter,1)
    print(f"RMSE = {rmse}")

    # Visualize the trajectories
    # vo = VisualOdometry(optim=optim)
    # visualize_trajectories(vo, iter)


    