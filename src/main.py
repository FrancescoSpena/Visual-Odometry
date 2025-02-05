import numpy as np
import VisualOdometry as vo
import utils as u
import matplotlib.pyplot as plt

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
    #world in camera frame
    T = v.cam.absolutePose()
    #camera in world frame and align with world camera
    T = u.alignWithWorldFrame(T)

    est_traj = []
    est_traj.append(np.eye(4))

    print(f"Status init: {status}")
    
    # print("-------------Estimation------------")
    # print(f"from frame {0} to frame {1}")
    # print(f"T_abs:\n {T}")
    # print("-------------Ground Truth----------")
    # T02 = u.g2T(gt[1])
    # print(f"T: \n {T02}")
    # print("\n")


    iter = 10
    for i in range(2, iter): 
        v.run(i)
        # world in camera frame 
        T_abs = v.cam.absolutePose()
        #camera in world frame and align with the world frame
        T_abs = u.alignWithWorldFrame(T_abs)
        
        est_traj.append(T_abs)

        # print("\n")
        # print("-------------Estimation------------")
        # print(f"from frame 0 to frame {i}")
        # print(f"T_abs:\n {T_abs}")
        # print("-------------Ground Truth----------")
        # T02 = u.g2T(gt[i])
        # print(f"T: \n {T02}")
        # print("\n")



    # est_traj = np.array(est_traj)
    # pos_est = np.array([T[:3, 3] for T in est_traj])
    
    gt_traj = []
    for i in range(0,iter-1):
        gt_traj.append(u.g2T(gt[i]))
    
    # gt_traj = np.array(gt_traj)
    # pos_gt = np.array([T[:3,3] for T in gt_traj])

    # plot(pos_gt, pos_est)



    #==============Evaluate=====================
    # #0_T_w
    # T0 = est_traj[0]
    # #1_T_w
    # T1 = est_traj[1]

    # gt0 = gt_traj[0]
    # gt1 = gt_traj[1]

    # #0_T_1
    # rel_T = T0 @ T1
    # rel_gt = np.linalg.inv(gt0) @ gt1

    # error_T = np.linalg.inv(rel_T) @ rel_gt

    # rot_part = np.trace(np.eye(3) - error_T[:3, :3])
    # norm = rel_T[:3, 3] / np.linalg.norm(rel_gt[:3, 3])

    # print(norm)


    for i in range(1,iter-1):
        print(f"from {i-1} to {i}")
        Ti = est_traj[i-1]
        Tnext_i = est_traj[i]

        gti = gt_traj[i-1]
        gtnext_i = gt_traj[i]

        rel_T = Ti @ Tnext_i
        rel_gt = np.linalg.inv(gti) @ gtnext_i

        error_T = np.linalg.inv(rel_T) @ rel_gt

        rot_part = np.trace(np.eye(3) - error_T[:3, :3])
        norm = np.linalg.norm(rel_T[:3, 3] / np.linalg.norm(rel_gt[:3, 3]))

        print(norm)
