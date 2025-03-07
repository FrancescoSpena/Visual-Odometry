import utils as u 
import numpy as np
import VisualOdometry as vo
import Camera as cam
import PICP_solver as s
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Main script Visual Odometry")
parser.add_argument('--iter', type=int, required=True, help="Number of iterations")
parser.add_argument('--picp', type=int, required=True, help="Number of iterations for icp")
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
    for elem in map: 
        id, point = elem
        project_point, isvalid = camera.project_point(point)

        if isvalid:
            valid +=1
            point = u.getPoint(point_curr_frame, str(id))
            if(point is not None):
                print(f"ID={id}, measure: {point}, proj: {project_point}")
        else:
            print(f"[test_proj]ID={id} No valid projection")
            no_valid +=1

    print(f"Projections valid: {valid}")
    print(f"Projections not valid: {no_valid}")
    print(f"Number of points in the map: {len(map)}")
    print("--Finish Data for this call--") 

    
def test_picp(camera, solver, map, points_frame, assoc, T_rel_gt=None, T_abs_gt=None):
    camera.updatePrev()
    iter_icp = args.picp 
    for _ in range(iter_icp):
        solver.initial_guess(camera, map, points_frame)
        solver.one_round(assoc)
        camera.updatePoseICP(solver.dx)
    
    #Estimated absolute pose
    T_abs = camera.absolutePose()
    T_abs_align = u.alignWithWorldFrame(T_abs)
    T_abs_align = np.round(T_abs_align, decimals=1)
    est_pose.append(T_abs_align)
    
    print(f"[test_picp]T_abs:\n {T_abs_align}")
    
    #GT absolute pose
    if(T_abs_gt is not None):
        print(f"[test_picp]T_abs_gt:\n {T_abs_gt}")
        
    camera.updateRelative()

    #Estimated relative pose
    print(f"[test_picp]T_rel:\n {np.round(camera.relativePose(), decimals=2)}")
    
    #GT relative pose
    if(T_rel_gt is not None):
        print(f"[test_picp]T_rel_gt:\n {np.round(T_rel_gt, decimals=2)}")
        

def updateMap(map, point_prev_frame, point_curr_frame, R, t, assoc, T_i):
    #map[i] = (ID, (x, y, z))
    #point_prev_frame[i] = (ID, (x, y))
    #point_curr_frame[i] = (ID, (x, y))
    #assoc[i] = (ID, best_ID)
    #T_i its the transformation 0_T_i

    id_in_map = [item[0] for item in map]
    id_curr_frame = [item[0] for item in point_curr_frame]
 
    #ID not in map
    missing = [item for item in id_curr_frame if item not in set(id_in_map)]
    
    if(len(missing) != 0):
        points_prev = []
        points_curr = []
        for elem in missing:
            point_prev = u.getPoint(point_prev_frame, str(elem))
            point_curr = u.getPoint(point_curr_frame, str(elem))

            if(point_prev is not None and point_curr is not None):
                points_prev.append(point_prev)
                points_curr.append(point_curr)
            else:
                print("[updateMap]Point_prev or Point_curr its None")
        
        missing_assoc = [(key, value) for key, value in assoc if key in missing]

        #New triangulated points w.r.t. frame i
        missing_map = u.triangulate(R, t, points_prev, points_curr, K, missing_assoc)

        #Take the missing map and transform all the point in the global frame
        transformed_missing_map = []
        T_i_inv = np.linalg.inv(T_i)
        for id, point in missing_map:
            point_global = transform_point(point, T_i_inv)
            transformed_missing_map.append((id, point_global))

        map.extend(transformed_missing_map)

def updateWithNewMeasurements(map, point_prev_frame, point_curr_frame, R, t, assoc, T_i):
    id_map = [item[0] for item in map]
    id_curr_frame = [item[0] for item in point_curr_frame]

    already_in_map = [item for item in id_curr_frame if item in set(id_map)]

    if(len(already_in_map) != 0):
        point_prev = []
        point_curr = []
        for elem in already_in_map:
            point_prev.append(u.getPoint(point_prev_frame, str(elem)))
            point_curr.append(u.getPoint(point_curr_frame, str(elem)))
        
        assoc_points = [(id, best) for id, best in assoc if id in already_in_map]

        #New points in frame i
        new_triangulation_map = u.triangulate(R, t, point_prev, point_curr, K, assoc_points)

        transformed_already_map = []
        T_i_inv = np.linalg.inv(T_i)
        for id, point in new_triangulation_map:
            point_global = transform_point(point, T_i_inv)
            transformed_already_map.append((id, point_global))
        
        for id, point in transformed_already_map:
            map = u.subPoint(map, id, point)



def process_frame(i, map, camera, solver, gt):
    path_frame_prev = u.generate_path(i)
    path_frame_curr = u.generate_path(i+1)

    data_frame_prev = u.extract_measurements(path_frame_prev)
    data_frame_curr = u.extract_measurements(path_frame_curr)

    _, _, points_prev, points_curr, assoc = u.data_association(data_frame_prev, data_frame_curr)
    
    
    #-------Relative Transformation gt-------
    T_prev = u.g2T(gt[i])   # frame i in world frame (w_T_i)
    T_curr = u.g2T(gt[i+1]) # frame i+1 in world frame (w_T_i+1)

    #w_T_i
    T_i = u.alignWithCameraFrame(T_prev.copy())
    T_i[np.abs(T_i) < 1e-2] = 0

    # T_i1 = u.alignWithCameraFrame(T_curr.copy())
    # T_i1[np.abs(T_i1) < 1e-2] = 0

    #i_T_i+1
    T_rel = np.linalg.inv(T_prev.copy()) @ T_curr.copy()
    T_rel = np.round(T_rel, decimals=2)
    T_align = u.alignWithCameraFrame(T_rel)

    # T_rel_est = camera.relativePose()
    # T_rel_est = np.round(T_rel_est, decimals=2)
    # R_curr, t_curr = u.T2m(T_rel_est)

    R_curr, t_curr = u.T2m(T_align)

    # R_curr[np.abs(R_curr) < 1e-2] = 0
    # t_curr[np.abs(t_curr) < 1e-2] = 0
    
    gt_pose.append(T_curr)


    #-------Relative Transformation gt-------

    #------Visual Odometry------
    #The absolute pose of the camera align with the frame i
    # T_i = camera.absolutePose()
    # T_i[np.abs(T_i) < 1e-2] = 0
    
    #------Visual Odometry------

    #-------Update with new measurements-------
    updateWithNewMeasurements(map, points_prev, points_curr, R_curr, t_curr, assoc, T_i)
    #-------Update with new measurements-------

    #-------PICP-------
    #points_curr are the points in the frame i+1
    test_picp(camera, solver, map, points_curr, assoc, T_rel_gt=T_align, T_abs_gt=T_curr)
    #-------PICP-------

    #-------Update Map-------
    updateMap(map, points_prev, points_curr, R_curr, t_curr, assoc, T_i)
    #-------Update Map-------

    


def main():
    path_frame0 = u.generate_path(0)
    path_frame1 = u.generate_path(1)

    data_frame0 = u.extract_measurements(path_frame0)
    data_frame1 = u.extract_measurements(path_frame1)

    p0, p1, points_frame0, points_frame1, assoc = u.data_association(data_frame0, data_frame1)
    
    #----------Complete VO-----------
    #Good rotation and translation is consistent to the movement (forward)
    
    # v = vo.VisualOdometry()
    # status = v.init()
    # print(f"Status: {status}")
    # T = v.cam.absolutePose()
    # R, t = u.T2m(T)

    #print(f"R:\n {R}, \nt:\n {t}")
    
    #----------Complete VO-----------

    #----------GT-----------
    T0_gt = u.g2T(gt[0])  # frame 0 in world frame (w_T_0)
    T1_gt = u.g2T(gt[1])  # frame 1 in world frame (w_T_1)

    # Compute relative pose: 0_T_1 = 0_T_w @ w_T_1
    T_rel = np.linalg.inv(T0_gt) @ T1_gt
    T_align = u.alignWithCameraFrame(T_rel)
    R, t = u.T2m(T_align)

    #----------GT-----------

    gt_pose.append(T0_gt)
    gt_pose.append(T1_gt)
    est_pose.append(T0_gt)

    # Triangulate points w.r.t. frame 0
    map = u.triangulate(R, t, p0, p1, K, assoc)

    print("Frame 0:")
    test_proj(map, points_frame0, camera)
    
    print("P-ICP")
    test_picp(camera, solver, map, points_frame1, assoc)

    print("Frame 1:")
    test_proj(map, points_frame1, camera)
    
    iter = args.iter
    if(iter > 120): 
        iter = 120
    
    for i in range(1, iter):
        print(f"From frame {i} to {i+1}")
        process_frame(i=i, map=map, camera=camera, solver=solver, gt=gt)
        
    print(f"Num. of est pose: {len(est_pose)}")
    print(f"Num. of gt pose: {len(gt_pose)}")
    plot(gt_pose, est_pose)

    #evaluate(est_pose, gt_pose)

        
if __name__ == '__main__':
    main()