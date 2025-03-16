import numpy as np
import matplotlib.pyplot as plt
import utils as u

camera_info = u.extract_camera_data()
K = camera_info['camera_matrix']
T_cam = camera_info['cam_transform']

def transform_point(p_cam, T):
    """
    Transform the point in a different reference frame
    """
    p_homog = np.append(p_cam, 1.0)
    p_world_homog = T @ p_homog
    p_world_homog = np.array(p_world_homog, dtype=np.float32)
    return p_world_homog[:3]
 

def versus(map, world):
    """
    Compare the map with the real map (world)
    """
    out = 0
    for elem in map:
        id, pos_cam = elem

        pos_cam = transform_point(pos_cam, T_cam)
        pos_world_gt = world[str(id)]['position']

        if(pos_cam[2] <= 0):
            out += 1

            
        print(f"ID={id} --> pos_est: {pos_cam}")
        print(f"ID={id} --> pos_gt: {pos_world_gt}")
        print("----")
        
    return out
    
def test_proj(map, point_curr_frame, camera):
    print("--Start Data for this call--")
    total_distance = 0
    for elem in map: 
        id, point, app = elem
        #Project the 3D point into the image plane
        project_point, isvalid = camera.project_point(point)

        if isvalid:
            point = u.getPoint(point_curr_frame, str(id))
            if(point is not None):
                #Compute the distance from the projected point and the real measure
                curr_dist = np.round(np.linalg.norm(np.array(point) - np.array(project_point)), decimals=2)
                total_distance += curr_dist
        else:
            print(f"[test_proj]ID={id} No valid projection")

    print(f"Total error distance: {total_distance}")
    print("--Finish Data for this call--") 

def plot_vectors(vec1, vec2):
    fig, ax = plt.subplots()
    
    # Extract points and IDs
    ids1, points1 = zip(*[(v[0], v[1]) for v in vec1])
    ids2, points2 = zip(*[(v[0], v[1]) for v in vec2])
    
    x1, y1 = zip(*points1)
    x2, y2 = zip(*points2)
    
    # Plot true points
    ax.scatter(x1, y1, c='b', marker='o', label='True Points')
    
    # Plot estimated points
    ax.scatter(x2, y2, c='r', marker='^', label='Estimated Points')
    
    # Add labels
    for i, (id1, id2) in enumerate(zip(ids1, ids2)):
        ax.text(x1[i], y1[i], f'{id1}', color='blue')
        ax.text(x2[i], y2[i], f'{id2}', color='red')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.show()

def plot(gt_traj, est_traj, map_points=None):
    """
    Plot the est and gt poses
    """
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
    ax.set_title("Ground Truth vs Estimated Trajectories")
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True)
    plt.show()

def evaluate(est_traj, gt_traj):
    """
    Compute the rot and tran error of the estimated poses
    """
    assert len(est_traj) == len(gt_traj)
    ratio = []
    out = 0
    for i in range(len(est_traj)-1):
        Ti_est = est_traj[i]
        Ti1_est = est_traj[i+1]
        Ti = gt_traj[i]
        Ti1 = gt_traj[i+1]

        rel_T = np.linalg.inv(Ti) @ Ti1
        rel_T_est = np.linalg.inv(Ti_est) @ Ti1_est

        error_T = np.linalg.inv(rel_T_est) @ rel_T

        tran_est_T = rel_T_est[:3, 3]
        tran_T = rel_T[:3, 3]
        
        norm_est_T = np.linalg.norm(tran_est_T)
        norm_T = np.linalg.norm(tran_T)

 
        if norm_T != 0:
            tran_error = norm_est_T / norm_T
            tran_error = np.round(tran_error, decimals=2)
            ratio.append(tran_error)
            # if(tran_error > 5.5):
            #     print(f"i: {i} --> tran error: {tran_error} Out of range")
            #     out+=1
            # else:
            #     print(f"i: {i} --> tran error: {tran_error}")

    scale_ratio = np.median(ratio)
    return scale_ratio

def scale_gt_poses(gt_pose, scale_ratio):
    """
    Scale the gt poses with the scale_ratio
    """
    scaled_gt_poses = []

    for pose in gt_pose:
        scaled_T = pose.copy()
        scaled_T[:3,3] *= scale_ratio
        scaled_gt_poses.append(scaled_T)
    
    return scaled_gt_poses
 
def scale_est_poses(est_pose, scale_ratio):
    """
    Scale the estimated poses with the scale_ratio
    """
    scaled_est_poses = []

    for pose in est_pose:
        scaled_T = pose.copy()
        scaled_T[:3,3] *= (1/scale_ratio)
        scaled_est_poses.append(scaled_T)
    
    return scaled_est_poses
