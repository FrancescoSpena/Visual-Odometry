import utils as u 
import numpy as np
import Camera as cam
import PICP_solver as s
import argparse
import matplotlib.pyplot as plt

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

#------For testing------
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
    assert len(est_traj) == len(gt_traj)
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
            if(tran_error > 1.5):
                print(f"i: {i} --> tran error: {tran_error} Out of range")
                out+=1
            else:
                print(f"i: {i} --> tran error: {tran_error}")

    print(f"total poses: {len(est_traj)-1}")
    print(f"out of range: {out}")
 
def transform_point(p_cam, T):
    p_homog = np.append(p_cam, 1.0)
    p_world_homog = T @ p_homog
    p_world_homog = np.array(p_world_homog, dtype=np.float32)
    return p_world_homog[:3]
    
def versus(map, world):
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
    valid = 0
    no_valid = 0
    total_distance = 0
    for elem in map: 
        id, point = elem
        project_point, isvalid = camera.project_point(point)

        if isvalid:
            valid +=1
            point = u.getPoint(point_curr_frame, str(id))
            if(point is not None):
                #print(f"ID={id}, measure: {point}, proj: {project_point}")
                curr_dist = np.round(np.linalg.norm(np.array(point) - np.array(project_point)), decimals=1)
                total_distance += curr_dist
                #print(f"distance: {curr_dist}")
        else:
            #print(f"[test_proj]ID={id} No valid projection")
            no_valid +=1

    #print(f"Number of points in the map: {len(map)}")
    print(f"Total error distance: {total_distance}")
    print("--Finish Data for this call--") 
#------For testing------
    
def picp(map, points_curr, camera, assoc_3d, i):
    iter_picp = args.picp
    for _ in range(iter_picp):
        solver.initial_guess(camera, map, points_curr)
        solver.one_round(assoc_3d)
        camera.updatePoseICP(solver.dx)
    
    T_abs_est = camera.absolutePose().copy()
    T_abs_est = u.alignWithWorldFrame(T_abs_est)
    T_abs_est = np.round(T_abs_est, decimals=1)
    T_abs_est[np.abs(T_abs_est) < 1e-1] = 0

    T_abs_gt = u.g2T(gt[i+1])
    T_abs_gt = np.round(T_abs_gt, decimals=1)
    T_abs_gt[np.abs(T_abs_gt) < 1e-1] = 0

    gt_pose.append(T_abs_gt)
    est_pose.append(T_abs_est)
    
    print(f"[process_frame]T_abs_est:\n {T_abs_est}")
    print(f"[process_frame]T_abs_gt:\n {T_abs_gt}")      


def updateMap(map, measurements_prev, measurements_curr, R, t, T_i):
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]

    missing = [item for item in id_curr if item not in set(id_map)]

    if(len(missing) != 0):
        prev_points = []
        curr_points = []
        assoc = []
        for elem in missing:
            prev = u.getPoint(measurements_prev, elem)
            curr = u.getPoint(measurements_curr, elem)

            if(prev is not None and curr is not None):
                prev_points.append(prev)
                curr_points.append(curr)
                assoc.append((elem, elem))
        
        missing_map = u.triangulate(R, t, prev_points, curr_points, K, assoc)

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
        for id in already_in_map:
            prev = u.getPoint(measurements_prev, str(id))
            curr = u.getPoint(measurements_curr, str(id))

            if prev is not None and curr is not None:
                prev_points.append(prev)
                curr_points.append(curr)
                assoc.append((id, id))
        
        R = np.round(R, decimals=2)
        t = np.round(t, decimals=2)

        # New points w.r.t. the frame i
        new_triangulation = u.triangulate(R, t, prev_points, curr_points, K, assoc)


        # Transform all points w.r.t global frame 
        if len(new_triangulation) != 0:
            transformed_map = []
            T_i_inv = np.linalg.inv(T_i)
            for id, point in new_triangulation:
                point_global = transform_point(point, T_i_inv)
                #print(f"id: {id} --> point global: {point_global}")
                transformed_map.append((str(id), point_global))

            for id, point in transformed_map:
                map = u.subPoint(map, id, point)

            print("[Retriangulation]All done!")
    else:
        print("[Retriangulation]No update!")    

    return map
    


def process_frame(i, map):
    print(f"From frame {i} to {i+1}")
    path_frame_prev = u.generate_path(i)
    path_frame_curr = u.generate_path(i+1)

    data_frame_prev = u.extract_measurements(path_frame_prev)
    data_frame_curr = u.extract_measurements(path_frame_curr)

    _, _, points_prev, points_curr, _ = u.data_association(data_frame_prev, data_frame_curr)


    T_prev = u.g2T(gt[i])
    T_curr = u.g2T(gt[i+1])

    T_i = u.alignWithCameraFrame(T_prev.copy())
    T_i[np.abs(T_i) < 1e-2] = 0

    T_rel = np.linalg.inv(T_prev.copy()) @ T_curr.copy()
    T_rel = u.alignWithCameraFrame(T_rel)
    T_rel = np.round(T_rel, decimals=2)
    R, t = u.T2m(T_rel)


    #------Retriangulation------
    map = retriangulation(map, points_prev, points_curr, R, t, T_i)
    #------Retriangulation------

    #------Update map------
    map = updateMap(map, points_prev, points_curr, R, t, T_i)
    #------Update map------
 
    #------PICP------
    assoc_3d = u.association3d(map, points_curr, camera)
    picp(map, points_curr, camera, assoc_3d, i)
    #------PICP------

    test_proj(map, points_curr, camera)
    

def main():
    print("From frame 0 to 1")
    path_frame0 = u.generate_path(0)
    path_frame1 = u.generate_path(1)

    data_frame0 = u.extract_measurements(path_frame0)
    data_frame1 = u.extract_measurements(path_frame1)

    p0, p1, _, points_frame1, assoc = u.data_association(data_frame0, data_frame1)
    
    #----------Complete VO-----------
    #Good rotation and translation is consistent to the movement (forward)
    
    # R, t = u.compute_pose(points_frame0, points_frame1, K=K)
    # R[np.abs(R) < 1e-1] = 0
    # t[np.abs(t) < 1e-1] = 0
    # camera.setCameraPose(u.m2T(R, t))


    #----------Complete VO-----------

    #----------GT-----------
    T0_gt = u.g2T(gt[0])  # frame 0 in world frame (w_T_0)
    T1_gt = u.g2T(gt[1])  # frame 1 in world frame (w_T_1)

    T_rel_gt = u.relativeMotion(T0_gt, T1_gt)
    T_rel_gt = u.alignWithCameraFrame(T_rel_gt)
    R, t = u.T2m(T_rel_gt)

    #----------GT-----------

    gt_pose.append(T0_gt)
    gt_pose.append(T1_gt)
    est_pose.append(T0_gt)

    # Triangulate points w.r.t. frame 0
    map = u.triangulate(R, t, p0, p1, K, assoc)
    camera.setCameraPose(u.m2T(R, t))

    T_abs_est = camera.absolutePose().copy()
    T_abs_est = u.alignWithWorldFrame(T_abs_est)
    T_abs_est = np.round(T_abs_est, decimals=1)
    T_abs_est[np.abs(T_abs_est) < 1e-1] = 0

    T1_gt = np.round(T1_gt, decimals=1)
    T1_gt[np.abs(T1_gt) < 1e-1] = 0

    print(f"[main]T_abs_est:\n {T_abs_est}")
    print(f"[main]T_abs_gt:\n {T1_gt}")


    iter = args.iter
    if(iter > 120):
        iter = 120
    for i in range(1, iter):
        #update the pose of the camera from frame i to i+1
        process_frame(i, map)

    if(args.plot):
        plot(gt_pose, est_pose)

if __name__ == '__main__':
    main()






