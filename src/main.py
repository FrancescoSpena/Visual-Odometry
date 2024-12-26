import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

def visualize_poses(gt_poses, estimated_poses):
    rr.init("pose_visualization", spawn=True)
    rr.log("estimated_trajectory", rr.Points3D(estimated_poses, colors=(255, 0, 0), radii=0.02))
    rr.log("ground_truth_trajectory", rr.Points3D(gt_poses, colors=(0, 255, 0), radii=0.02))

def evaluate(T_curr, T_next, T_gt_curr, T_gt_next):
    rel_T = np.linalg.inv(T_curr) @ T_next
    rel_gt = np.linalg.inv(T_gt_curr) @ T_gt_next
    error_T = np.linalg.inv(rel_T) @ rel_gt
    rot_part = np.trace(np.eye(3) - error_T[0:3, 0:3])
    t_part = np.linalg.norm(rel_T[0:3, 3]) / np.linalg.norm(rel_gt[0:3, 3])

    print(f"rotation: {int(rot_part)}")
    print(f"translation: {t_part}")
    print("======")


if __name__ == "__main__":
    v = vo.VisualOdometry()
    estimated = []
    R, t, scale, points_3d, status = v.init()
    estimated.append((R,t))

    # print(f"R: {R}")
    # print(f"t: {t}")
    # print(f"scale: {scale}")
    # print(f"first point 3D: {points_3d[0]}")
    print(f"Status init: {status}")

    gt = v.traj
    

    for i in range(1, 10):
        R, t, R_rel, t_rel, _ = v.run(i)
        estimated.append((R_rel,t_rel))
    
    print("Starting evaluation...")
    for i in range(len(estimated)-1):
        T_curr = u.m2T(estimated[i][0], estimated[i][1])
        T_next = u.m2T(estimated[i+1][0], estimated[i+1][1])
        T_gt_curr = u.gt2T(gt[i])
        T_gt_next = u.gt2T(gt[i+1])

        # print(f"T_curr: {T_curr}")
        # print(f"T_next: {T_curr}")
        # print(f"T_gt_curr: {T_gt_curr}")
        # print(f"T_gt_next: {T_gt_next}")

        evaluate(T_curr, T_next, T_gt_curr, T_gt_next)

