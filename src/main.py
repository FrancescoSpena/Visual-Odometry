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


def main():
    gt = u.read_traj()

    gt_traj = []
    est_traj = []

    v = vo.VisualOdometry()
    #world in frame 0 (estimated)
    T = v.cam.absolutePose()
    #frame 0 in world (estimated)
    T = u.alignWithWorldFrame(T)
    #frame 0 in world (ground truth)
    gt0 = u.g2T(gt[0])

    gt_traj.append(gt0)
    est_traj.append(T)
    
    #inizialize the Visual Odometry system
    status = v.init()
    print(f"Status system: {status}")
    
    #world in frame 1 (estimated)
    T = v.cam.absolutePose()
    #frame 1 in world (estimated)
    T = u.alignWithWorldFrame(T)
    T = np.round(T)
    #frame 1 in world (ground truth)
    gt1 = u.g2T(gt[1])

    gt_traj.append(gt1)
    est_traj.append(T)

    # print("frame 1 in world:")
    # print(f"T:\n {est_traj[1]},\n gt:\n {gt_traj[1]}")

    print("Compute...")
    print("----------------------")
    iter = 50
    for i in range(2, iter):
        #print(f"Update pose with frame {i}")
        v.run(i)
        #world in frame i
        T = v.cam.absolutePose()
        #frame i in world
        T = u.alignWithWorldFrame(T)
        T = np.round(T)
        gti = u.g2T(gt[i])

        gt_traj.append(gti)
        est_traj.append(T)
    
    for i in range(0, iter):
        print(f"frame {i} in world:")
        print(f"est:\n {est_traj[i]},\n gt:\n {gt_traj[i]}")
        print("----------------------")

    est_traj = np.array(est_traj)
    pos_est = np.array([T[:3, 3] for T in est_traj])

    gt_traj = np.array(gt_traj)
    pos_gt = np.array([T[:3, 3] for T in gt_traj])

    plot(pos_gt, pos_est)


if __name__ == "__main__":
    main()
