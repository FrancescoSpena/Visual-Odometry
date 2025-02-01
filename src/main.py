import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def evaluate(pos_gt, pos_est):
    pass

def plot(pos_gt, pos_est):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], 'bo-', markersize=5, label="Ground Truth")
    ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2], 'ro-', markersize=5, label="Estimated")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Truth vs Estimated")
    ax.legend()
    plt.grid()
    plt.show()

def plotGt(pos_gt):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], 'bo-', markersize=5, label="Ground Truth")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Truth Trajectory")
    ax.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    gt = u.read_traj()
    estimated_pose = []

    v = vo.VisualOdometry()
    status = v.init()
    T = v.cam.absolutePose()


    est_traj = []
    est_traj.append(np.eye(4))

    print(f"Status init: {status}")
    
    print("-------------Estimation------------")
    print(f"from frame {0} to frame {1}")
    print(f"T_abs:\n {T}")
    print(f"from frame {0} to frame {1}")
    print(f"T_rel:\n {v.cam.relativePose()}")
    print("-------------Ground Truth----------")
    T_gt_rel = u.relativeMotion(u.g2T(gt[0]), u.g2T(gt[1]))
    print(f"T: \n {T_gt_rel}")
    print("\n")


    iter = 10
    for i in range(2, iter): 
        v.run(i)
        T_abs = v.cam.absolutePose()
        est_traj.append(T_abs)

        print("\n")
        print("-------------Estimation------------")
        print(f"from frame 0 to frame {i}")
        print(f"T_abs:\n {T_abs}")
        print(f"from frame {i-1} to frame {i}")
        print(f"T_rel:\n {v.cam.relativePose()}")
        print("-------------Ground Truth----------")
        T02 = u.g2T(gt[i])
        print(f"T: \n {T02}")
        T_gt_rel = u.relativeMotion(u.g2T(gt[i-1]), u.g2T(gt[i]))
        print(f"T_rel_gt:\n {T_gt_rel}")
        print("\n")

    # est_traj.append(T)

    # new_est = []
    # for i in range(0,len(est_traj)):
    #     Ti = est_traj[i]
    #     Ti[2, 3] = 0
    #     Ti[3, 3] = 0
    #     new_est.append(Ti)


    # est_traj = np.array(new_est)
    # pos_est = np.array([T[:3, 3] for T in est_traj])
    

    # # plot(pos_gt, pos_est)

    # gt_traj = []
    # for i in range(0,len(gt)):
    #     gt_traj.append(u.g2T(gt[i]))
    
    # gt_traj = np.array(gt_traj)
    # pos_gt = np.array([T[:3,3] for T in gt_traj])

    # plot(pos_gt, pos_est)



    #==============Evaluate=====================
    # T0 = est_traj[0]
    # T1 = est_traj[1]

    # gt0 = gt_traj[0]
    # gt1 = gt_traj[1]

    # rel_T = np.linalg.inv(T0) @ T1
    # rel_gt = np.linalg.inv(gt0) @ gt1

    # error_T = np.linalg.inv(rel_T) @ rel_gt

    # rot_part = np.trace(np.eye(3) - error_T[:3, :3])
    # norm = rel_T[:3, 3] / np.linalg.norm(rel_gt[:3, 3])

    # print(norm)