import utils as u 
import numpy as np
import Camera as cam
import PICP_solver as s
import argparse
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description="Main script Visual Odometry")
parser.add_argument('--iter', type=int, required=True, help="Number of iterations")
parser.add_argument('--picp', type=int, required=True, help="Number of iterations for icp")
parser.add_argument('--plot', type=bool, required=False, help="Active Plot", default=False)
args = parser.parse_args()

#Camera
camera_info = u.extract_camera_data()
K = camera_info['camera_matrix']
T_cam = camera_info['cam_transform']

#World
world_info = u.extract_world_data()

#Trajectory
gt = u.read_traj()

#P-ICP
camera = cam.Camera(K)
solver = s.PICP(camera=camera)

gt_pose = []
est_pose = []
pose_for_track = []
points_track = {}

#------For testing------
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
            if(tran_error > 5.5):
                print(f"i: {i} --> tran error: {tran_error} Out of range")
                out+=1
            else:
                print(f"i: {i} --> tran error: {tran_error}")

    print(f"total poses: {len(est_traj)-1}")
    print(f"out of range: {out}")

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
        id, point = elem
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
#------For testing------

#------Multi View------
def triangulate_n_views(point_tracks, projection_matrices):
    """
    Triangulate a 3D point from multi view using DLT method

    Args:
        point_tracks (dict): { frame_id: (x, y) } 2D points observed in every frame
        projection_matrices (dict): { frame_id: P } projection matrix 3x4 for every frame 

    Returns:
        np.array: 3D coordinates of the reconstructed point [X, Y, Z]
    """
    A = []

    for frame_id, (x, y) in point_tracks.items():
        x, y = float(x), float(y)
        
        if frame_id not in projection_matrices:
            raise ValueError(f"Frame ID {frame_id} not found in projection_matrices.")
        
        P = projection_matrices[frame_id]  # Projection matrix 3x4

        A.append(x * P[2] - P[0]) 
        A.append(y * P[2] - P[1])  

    A = np.array(A) 
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]  
    
    if X[3] == 0:
        raise ValueError("Homogeneous coordinate X[3] is zero. Cannot convert to Cartesian coordinates.")
    
    return X[:3] / X[3]

def extract_measurements_by_id(points_track, id_point):
    """
    Extract the dict that represent the track of the point with id=id_point
    Args: 
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        id_point (str): id of the point
    Return:
        result (dict): that represent the track of the point with id=id_point

    """
    result = {}
    for frame_id, points in points_track.items():
        for point_id, coordinates in points:
            if point_id == id_point:
                result[frame_id] = coordinates
    return result
    
def add_point_to_frame(points_track, frame_id, point_id, point):
    """
    Update the dictionary points_track with the new measure

    Args:
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        frame_id (int): id of the frame
        point_id (str): id of the point
        point (np.array): measure of the point with id=point_id
  
    """
    if frame_id not in points_track:
        points_track[frame_id] = []
        
    point = np.array(point, dtype=np.float32)
    points_track[frame_id].append((point_id, point))

#------Multi View------

def picp(map, points_curr, camera, assoc_3d, i):
    #Apply P-ICP to align the camera with the curr frame
    iter_picp = args.picp
    for _ in range(iter_picp):
        solver.initial_guess(camera, map, points_curr)
        solver.one_round(assoc_3d)
        camera.updatePoseICP(solver.dx)
    
    #Update the pose_to_track list for the triangulation in multi views
    T_abs_est = camera.absolutePose().copy()
    pose_for_track.append(T_abs_est)
   
    #------Update relative pose------
    T_rel_est = camera.updateRelative(T_abs_est).copy()
    T_rel_est = u.alignWithWorldFrame(T_rel_est)
    T_rel_est = np.round(T_rel_est, decimals=2)
    #------Update relative pose------

    #Align the pose with the world frame (for the plot)
    T_abs_est = u.alignWithWorldFrame(T_abs_est)

    #Compute the gt absolute pose
    T_abs_gt = u.g2T(gt[i+1])
    T_abs_gt = np.round(T_abs_gt, decimals=1)
    T_abs_gt[np.abs(T_abs_gt) < 1e-1] = 0

    #Compute the gt relative pose
    T_rel_gt = np.linalg.inv(u.g2T(gt[i])) @ T_abs_gt
    T_rel_gt = np.round(T_rel_gt, decimals=1)
    T_rel_gt[np.abs(T_rel_gt) < 1e-1] = 0

    #Append in the right lists
    gt_pose.append(T_abs_gt)
    est_pose.append(T_abs_est)
    
    #------For printing------
    print(f"[process_frame]T_abs_est:\n {T_abs_est}")
    print(f"[process_frame]T_abs_gt:\n {T_abs_gt}")    
    print(f"[process_frame]T_rel_est:\n {T_rel_est}")
    print(f"[process_frame]T_rel_gt:\n {T_rel_gt}")
    #------For printing------  


def updateMap(map, measurements_prev, measurements_curr, R, t, T_i):
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]

    #ID of the new no mapped points
    missing = [item for item in id_curr if item not in set(id_map)]

    if(len(missing) != 0):
        #Recover the measure of the prev and curr frame
        prev_points = []
        curr_points = []
        assoc = []
        for elem in missing:
            prev = u.getPoint(measurements_prev, str(elem))
            curr = u.getPoint(measurements_curr, str(elem))

            if(prev is not None and curr is not None):
                prev_points.append(prev)
                curr_points.append(curr)
                assoc.append((elem, elem))
        
        #Triangulation of the missing points w.r.t. the prev frame
        missing_map = u.triangulate(R, t, prev_points, curr_points, K, assoc)

        #Report the points in the global frame and extend the map
        if(len(missing_map) != 0):
            transformed_map = []
            T_i_inv = np.linalg.inv(T_i)

            for id, point in missing_map:
                point_global = transform_point(point, T_i_inv)
                transformed_map.append((id, point_global))
    
            map.extend(transformed_map)

            print("[updateMap]All done!")
    
    return map

def retriangulation(map, measurements_prev, measurements_curr, R, t, T_i):
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]

    already_in_map = [item for item in id_curr if item in set(id_map)]

    prev_points = []
    curr_points = []
    assoc = []

    if len(already_in_map) != 0:
        #Recover the measure in the prev and curr frame of the point already in map
        for id in already_in_map:
            prev = u.getPoint(measurements_prev, str(id))
            curr = u.getPoint(measurements_curr, str(id))

            if prev is not None and curr is not None:
                prev_points.append(prev)
                curr_points.append(curr)
                assoc.append((id, id))
        
        # New points w.r.t. the frame i
        new_triangulation = u.triangulate(R, t, prev_points, curr_points, K, assoc)

        # Transform all points w.r.t global frame 
        if len(new_triangulation) != 0:
            transformed_map = []
            T_i_inv = np.linalg.inv(T_i)
            for id, point in new_triangulation:
                point_global = transform_point(point, T_i_inv)
                transformed_map.append((str(id), point_global))

            for id, point in transformed_map:
                map = u.subPoint(map, id, point)

            print("[Retriangulation]All done!")
    return map
    
def retriangulation_n_views(map, est_pose, track, measurements_curr):
    """
    Args:
        map (list): {id, (x, y, z)}: 3D points 
        est_pose (list): {pose0, pose1, ...} a list of absolute pose
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        measurements_curr (list): (id, (x,y)) list of measurements
        id_frame: id of the frame
    Returns:
        map: the updated map with new triangulated points
    """

    #Reconstruct the projection matrices from the pose and build the dict
    projection_matrices = []
    for pose in est_pose:
        R, t = u.T2m(pose)
        R = np.round(R, 2)
        t = np.round(t, 2)
        P_curr = K @ np.hstack((R,t))
        projection_matrices.append(P_curr)
    
    dict_projection_matrices = {}
    for i, proj in enumerate(projection_matrices):
        dict_projection_matrices[i] = proj


    #Recover the id of the points to update
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]
    already_in_map = [item for item in id_curr if item in set(id_map)]

    if(len(already_in_map) != 0):
        #Retriangulation of the point with multi view and update the map with the new points
        for id in already_in_map:
            res = extract_measurements_by_id(track, str(id))
            if res:
                new_point = triangulate_n_views(res, dict_projection_matrices) 
                if(new_point is not None):
                    map = u.subPoint(map, id, new_point) 
            else:
                print("no point found")
    
    print("[Retriangulation]All done!")
    return map


        
def process_frame(i, map):
    print(f"From frame {i} to {i+1}")
    path_frame_prev = u.generate_path(i)
    path_frame_curr = u.generate_path(i+1)

    data_frame_prev = u.extract_measurements(path_frame_prev)
    data_frame_curr = u.extract_measurements(path_frame_curr)

    points_prev, points_curr = u.getMeasurementsFromDataFrame(data_frame_prev, data_frame_curr)

    T_i = camera.absolutePose().copy()
    #T_i[np.abs(T_i) < 1e-1] = 0

    T_rel_est = camera.relativePose().copy()
    #T_rel_est = np.round(T_rel_est, 3)
    #T_rel_est[np.abs(T_rel_est) < 1e-1] = 0 
    R, t = u.T2m(T_rel_est)

    #------Retriangulation------
    map = retriangulation_n_views(map=map, 
                                  est_pose=pose_for_track, 
                                  track=points_track, 
                                  measurements_curr=points_curr)
    # #------Retriangulation------

    #------Update map------
    map = updateMap(map, points_prev, points_curr, R, t, T_i)
    #------Update map------
 
    #------PICP------
    assoc_3d = u.association3d(map, points_curr, camera)
    picp(map, points_curr, camera, assoc_3d, i)
    #------PICP------

    for id, point in points_curr:
        add_point_to_frame(points_track=points_track, frame_id=i+1, point_id=id, point=point)

    #test_proj(map, points_curr, camera)
    

def main():
    print("From frame 0 to 1")
    path_frame0 = u.generate_path(0)
    path_frame1 = u.generate_path(1)

    data_frame0 = u.extract_measurements(path_frame0)
    data_frame1 = u.extract_measurements(path_frame1)

    p0, p1, points_frame0, points_frame1, assoc = u.data_association(data_frame0, data_frame1)
    
    for id, point in points_frame0:
        add_point_to_frame(points_track=points_track, frame_id=0, point_id=id, point=point)
    
    for id, point in points_frame1:
        add_point_to_frame(points_track=points_track, frame_id=1, point_id=id, point=point)


    #----------Complete VO-----------
    #Good rotation and translation is consistent to the movement (forward)
    
    R, t = u.compute_pose(points_frame0, points_frame1, K=K)
    #R = np.round(R, decimals=2)
    #t = np.round(t, decimals=2)
    # print(f"R:\n {R}")
    # print(f"t:\n {t}")
    T1_est = u.m2T(R,t)
    #T1_est = np.round(T1_est, 2)


    #----------Complete VO-----------

    #----------GT-----------
    T0_gt = u.g2T(gt[0])  # frame 0 in world frame (w_T_0)
    T1_gt = u.g2T(gt[1])  # frame 1 in world frame (w_T_1)

    T_rel_gt = u.relativeMotion(T0_gt, T1_gt)
    T_rel_gt = np.round(T_rel_gt, 2)
    #R, t = u.T2m(T_rel_gt)

    #----------GT-----------

    gt_pose.append(T0_gt)
    gt_pose.append(T1_gt)

    # Triangulate points w.r.t. frame 0
    map = u.triangulate(R, t, p0, p1, K, assoc)
    camera.setCameraPose(T1_est)

    #test_proj(map, points_frame1, camera)
    
    #Pose align w.r.t. world frame
    est_pose.append(T0_gt)
    est_pose.append(u.alignWithWorldFrame(T1_est))

    #[print(elem) for elem in est_pose]

    #The pose in this list its align w.r.t. camera frame
    pose_for_track.append(np.eye(4)) 
    pose_for_track.append(T1_est)

    T_abs_est = camera.absolutePose().copy()
    T_abs_est = u.alignWithWorldFrame(T_abs_est)
    T_abs_est = np.round(T_abs_est, decimals=1)
    T_abs_est[np.abs(T_abs_est) < 1e-1] = 0

    T1_gt = np.round(T1_gt, decimals=1)
    T1_gt[np.abs(T1_gt) < 1e-1] = 0

    T_rel_est = camera.relativePose().copy()
    T_rel_est = u.alignWithWorldFrame(T_rel_est)
    T_rel_est = np.round(T_rel_est, decimals=1)
    T_rel_est[np.abs(T_rel_est) < 1e-1] = 0

    print(f"[main]T_abs_est:\n {T_abs_est}")
    print(f"[main]T_abs_gt:\n {T1_gt}")
    print(f"[main]T_rel_est:\n {T_rel_est}")
    print(f"[main]T_rel_gt:\n {T_rel_gt}")


    iter = args.iter
    if(iter > 120):
        iter = 120
    for i in range(1, iter):
        #update the pose of the camera from frame i to i+1
        process_frame(i, map)


    ratio = evaluate(est_pose, gt_pose)
    scale_est = scale_est_poses(est_pose=est_pose, scale_ratio=ratio)

    if(args.plot):
        plot(gt_pose, scale_est)





if __name__ == '__main__':
    main()

