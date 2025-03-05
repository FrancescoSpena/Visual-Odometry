import utils as u 
import numpy as np
import VisualOdometry as vo
import Camera as cam
import PICP_solver as s
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Main script Visual Odometry")
parser.add_argument('--iter', type=int, required=True, help="Number of iterations")
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


def transform_point(p_cam, T):
    p_homog = np.append(p_cam, 1.0)
        #w_p = w_T_c @ c_p
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
    
def test_proj(map, point_frame, camera):
    print("--Start Data for this call--")
    all_equal = 0
    all_good = 0
    i = 0
    for elem in map:
        id, point = elem
        project_point, isvalid = camera.project_point(point)
        
        if isvalid:
            point_true = u.getPoint(point_frame, id)
            if(point_true is not None):
                all_good +=1
                print(f"ID={id}, true point: {point_true}, proj: {project_point}")
        else:
            all_equal+=1
        i +=1

    print(f"Point valid: {all_good}")
    print(f"Point not valid: {all_equal}")
    print(f"Number of points in the map: {len(map)}")
    print("--Finish Data for this call--")  

    
def test_picp(camera, solver, map, points_frame, assoc, T_rel_gt=None, T_abs_gt=None):
    camera.updatePrev()
    for _ in range(7):
        solver.initial_guess(camera, map, points_frame)
        solver.one_round(assoc)
        camera.updatePoseICP(solver.dx)
    
    T_abs = camera.absolutePose()
    T_abs_align = u.alignWithWorldFrame(T_abs)
    est_pose.append(T_abs_align)
    print(f"T_abs:\n {np.round(T_abs_align, decimals=2)}")
    if(T_abs_gt is not None):
        print(f"T_abs_gt:\n {np.round(T_abs_gt, decimals=2)}")
    camera.updateRelative()
    print(f"T_rel:\n {np.round(camera.relativePose(), decimals=2)}")
    if(T_rel_gt is not None):
        print(f"T_rel_gt:\n {np.round(T_rel_gt, decimals=2)}")


def updateMap(map, point_prev_frame, point_curr_frame, R, t, assoc, camera, i):
    #map[i] = (ID, (x, y, z))
    #point_prev_frame[i] = (ID, (x, y))
    #point_curr_frame[i] = (ID, (x, y))
    #assoc[i] = (ID, best_ID)

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
        
        
        missing_assoc = [(key, value) for key, value in assoc if key in missing]
  
        #New triangulated points w.r.t. prev frame
        missing_map = u.triangulate(R, t, points_prev, points_curr, K, missing_assoc)

        #i+1_T_i
        T_rel_inv = np.linalg.inv(u.m2T(R, t))

        print(f"[UpdateMap]T_rel_inv:\n {T_rel_inv}")

        transformed_missing_map = []
        for id, point in missing_map:
            #print(f"------Start transform point with id: {id}------")
            point_h = np.append(point, 1)
            #print(f"point in prev frame: {point_h}")
                
            #i+1_p = i+1_T_i @ i_p
            point_transformed = T_rel_inv @ point_h

            #print(f"point in curr frame: {point_transformed}")
            transformed_missing_map.append((id, point_transformed[:3]))

        test_proj(transformed_missing_map, point_curr_frame, camera)
        
        map.extend(transformed_missing_map)


        print(f"[UpdateMap]Update the map with missing points, len map: {len(map)}")
    
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

    # print(T_rel)
    #T_align: transformation align with camera frame
    #c_T_w
    T_align = u.alignWithCameraFrame(T_rel)

    R, t = u.T2m(T_align)

    #print(f"R:\n {R}, \nt:\n {t}")

    #----------GT-----------

    gt_pose.append(T0_gt)
    gt_pose.append(T1_gt)
    est_pose.append(T0_gt)

    # Triangulate points w.r.t. frame 0
    map = u.triangulate(R, t, p0, p1, K, assoc)

    #versus(map, world_info)

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
        path_frame_prev = u.generate_path(i)
        path_frame_curr = u.generate_path(i+1)

        data_frame_prev = u.extract_measurements(path_frame_prev)
        data_frame_curr = u.extract_measurements(path_frame_curr)

        
        _, _, points_prev, points_curr, assoc = u.data_association(data_frame_prev, data_frame_curr)
        
        #------Relative transformation gt------
        T_prev = u.g2T(gt[i])   # frame i in world frame (w_T_i)
        T_curr = u.g2T(gt[i+1]) # frame i+1 in world frame (w_T_i+1)
        
        #i_T_i+1 = i_T_w @ w_T_i+1 = inv(w_T_i) @ w_T_i+1
        T_rel = np.linalg.inv(T_prev) @ T_curr
        T_align = u.alignWithCameraFrame(T_rel)

        R_curr, t_curr = u.T2m(T_align)

        gt_pose.append(T_curr)


        #------Relative transformation gt------

        print("[Main]P-ICP before update map")
        test_picp(camera, solver, map, points_curr, assoc, T_rel_gt=T_align, T_abs_gt=T_curr)

        #------Update the map------
        
        
        updateMap(map=map, 
                  point_prev_frame=points_prev, 
                  point_curr_frame=points_curr,
                  R=R_curr,
                  t=t_curr,
                  assoc=assoc,
                  camera=camera,
                  i=i)
        
        #------Update the map------

        
    print(f"Num. of est pose: {len(est_pose)}")
    print(f"Num. of gt pose: {len(gt_pose)}")
    plot(gt_pose, est_pose)

        
if __name__ == '__main__':
    main() 



'''

map (after update) = (map + missing_map)
camera = it's align with the frame i+1
camera_test = it's align with the frame i

test_proj(map, points_curr, camera) has the map correct but missing_map refered to the 
prev frame
test_proj(map, points_curr, camera_test) has the missing map correct but map refered to the
curr frame

'''