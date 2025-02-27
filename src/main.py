import numpy as np
import VisualOdometry as vo
import utils as u
import matplotlib.pyplot as plt
import time

def plot(gt_traj, est_traj, map_points=None):
    est_traj = np.array(est_traj)
    pos_est = np.array([T[:3, 3] for T in est_traj])

    gt_traj = np.array(gt_traj)
    pos_gt = np.array([T[:3, 3] for T in gt_traj])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2],
            'bo-', markersize=5, label="Ground Truth")
    ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2],
            'ro-', markersize=5, label="Estimated")

    if map_points is not None:
        map_xyz = np.array([point[1] for point in map_points])

        ax.scatter(map_xyz[:, 0], map_xyz[:, 1], map_xyz[:, 2],
                marker='^', color='g', s=50, label="Map Points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Ground Truth vs Estimated Trajectories with Map Points")
    ax.legend()
    plt.grid(True)
    plt.show()

def evaluate(est_traj, gt_traj):
    for i in range(0, len(est_traj)-1):
        Ti = est_traj[i]
        Ti_next = est_traj[i+1]

        Ti_gt = gt_traj[i]
        Ti_next_gt = gt_traj[i+1]

        T_rel = u.relativeMotion(Ti, Ti_next)
        T_rel_gt = u.relativeMotion(Ti_gt, Ti_next_gt)

        error_T = np.linalg.inv(T_rel) @ T_rel_gt
        
        rot_part = np.trace(np.eye(3) - error_T[:3, :3])
        tran_part = np.linalg.norm(T_rel[:3, 3]) / np.linalg.norm(T_rel_gt[:3, 3])
        

        print(f"frame {i}")
        #print(f"rot part: {rot_part}")
        print(f"tran part: {np.round(tran_part)}")
        print("-------------")
    

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
        print(f"[Main]Update pose with frame {i}")
        v.run(i)
        #world in frame i
        T = v.cam.absolutePose()
        #frame i in world
        T = u.alignWithWorldFrame(T)
        gti = u.g2T(gt[i])

        gt_traj.append(gti)
        est_traj.append(T)

        print("======================")
    
    plot(gt_traj, est_traj)
    #evaluate(est_traj, gt_traj)


if __name__ == "__main__":
    main()

    '''
    Il problema rimane sempre lo stesso. Dal frame 49 la matrice H è molto vicina ad 
    essere singolare e la stima va a caso. Bisogna capire il perchè. 
    '''