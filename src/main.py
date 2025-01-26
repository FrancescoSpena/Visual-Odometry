import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(pos_gt, pos_estimated):
    x_gt = [item[0] for item in pos_gt]
    y_gt = [item[1] for item in pos_gt]
    z_gt = [item[2] for item in pos_gt]

    x_est = [item[0, 0] for item in pos_estimated]
    y_est = [item[1, 0] for item in pos_estimated]
    z_est = [item[2, 0] for item in pos_estimated]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_gt, y_gt, z_gt, label='traj gt', marker='o')

    ax.plot(x_est, y_est, z_est, label='traj est', marker='x')

    ax.scatter(0, 0, 0, color='red', s=100, label='Origin')  # A red dot at the origin

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()



def evaluate(T_curr, T_next, T_gt_curr, T_gt_next):
    rel_T = np.linalg.inv(T_curr) @ T_next
    rel_gt = np.linalg.inv(T_gt_curr) @ T_gt_next
    error_T = np.linalg.inv(rel_T) @ rel_gt
    rot_part = np.trace(np.eye(3) - error_T[0:3, 0:3])
    t_part = np.linalg.norm(rel_T[0:3, 3]) / np.linalg.norm(rel_gt[0:3, 3])

    print(f"rotation: {rot_part}")
    print(f"translation: {int(t_part)}")
    print("======")


if __name__ == "__main__":
    gt = u.read_traj()
    estimated_pose = []

    v = vo.VisualOdometry()
    status = v.init()
    T = v.cam.worldInCameraPose()

    R, t = u.T2m(T)
    estimated_pose.append(t)

    print(f"Status init: {status}")
    print(f"from frame 0 to frame 1 -->\n estimated t:\n {t}\n gt t: {gt[0]}")

    iter = 2
    for i in range(1, iter):
        v.run(i)
        T = v.cam.worldInCameraPose()
        R, t = u.T2m(T)

        estimated_pose.append(t)
        
        print(f"from frame {i} to frame {i+1} -->\n estimated t:\n {t}\n gt t: {gt[i]}")

    visualize(gt[:iter], estimated_pose)