import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

def visualize_poses(gt_points, estimated_points):
    rr.init("pose_visualization", spawn=True)
    rr.log("estimated_trajectory", rr.Points3D(estimated_points, colors=(255, 0, 0), radii=0.02))
    rr.log("ground_truth_trajectory", rr.Points3D(gt_points, colors=(0, 255, 0), radii=0.02))

def evaluate(T_curr, T_next, T_gt_curr, T_gt_next):
    rel_T = np.linalg.inv(T_curr) @ T_next
    rel_gt = np.linalg.inv(T_gt_curr) @ T_gt_next
    error_T = np.linalg.inv(rel_T) @ rel_gt
    rot_part = np.trace(np.eye(3) - error_T[0:3, 0:3])
    t_part = np.linalg.norm(rel_T[0:3, 3]) / np.linalg.norm(rel_gt[0:3, 3])

    print(f"rotation: {int(rot_part)}")
    print(f"translation: {int(t_part)}")
    print("======")


if __name__ == "__main__":
    gt = u.read_traj()
    estimated_pose = []

    v = vo.VisualOdometry()
    T, points_3d, status = v.init()
    estimated_pose.append(T)

    print(f"Status init: {status}")

    iter = 100
    for i in range(1, iter):
        v.run(i)
        print(f"Update to frame {i}")
        T = v.cam.relative_pose()
        R, t = u.T2m(T)

        path_curr_gt = u.generate_path(i)
        path_next_gt = u.generate_path(i+1)
        info_gt = u.extract_other_info(path_curr_gt)
        info_gt_next = u.extract_other_info(path_next_gt)

        gt_0 = np.array(info_gt['Ground_Truths'])
        gt_1 = np.array(info_gt_next['Ground_Truths'])

        gt_dist = np.linalg.norm(gt_1 - gt_0)
        vo_dist = np.linalg.norm(t)
        scale = gt_dist / vo_dist

        t *= scale
        T = u.m2T(R, t)
        estimated_pose.append(T)


    print("Evaluation...")
    for i in range(0,iter-1):
        T_curr = estimated_pose[i]
        T_next = estimated_pose[i+1]
        T_gt_curr = u.gt2T(gt[i])
        T_gt_next = u.gt2T(gt[i+1])
        evaluate(T_curr, T_next, T_gt_curr, T_gt_next)


