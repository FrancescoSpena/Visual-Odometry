import utils as u 
import numpy as np
import VisualOdometry as vo
import Camera as cam
import PICP_solver as s

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


def testTransf():
    # Generate a random transformation matrix (rotation + translation)
    def random_transform():
        theta = np.random.uniform(0, 2*np.pi)
        R = u.Rz(theta) @ u.Ry(theta)
        t = np.random.randn(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    # Test the inverse relationship
    T_original = random_transform()

    # Test alignWithCameraFrame(alignWithWorldFrame(T))
    T_transformed = u.alignWithWorldFrame(T_original)
    T_recovered = u.alignWithCameraFrame(T_transformed)

    print("Original:\n", T_original)
    print("\nRecovered:\n", T_recovered)
    print("\nDifference:\n", T_original - T_recovered)
    print("\nIs close:", np.allclose(T_original, T_recovered, atol=1e-6))

    # Test alignWithWorldFrame(alignWithCameraFrame(T))
    T_transformed_back = u.alignWithCameraFrame(T_original)
    T_recovered_back = u.alignWithWorldFrame(T_transformed_back)

    print("\n--- Reverse Test ---")
    print("Original:\n", T_original)
    print("\nRecovered (reverse):\n", T_recovered_back)
    print("\nDifference (reverse):\n", T_original - T_recovered_back)
    print("\nIs close (reverse):", np.allclose(T_original, T_recovered_back, atol=1e-6))

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
    print("--Finish Data for this frame--")  

    
def test_picp(camera, solver, map, points_frame, assoc):
    camera.updatePrev()
    for _ in range(3):
        solver.initial_guess(camera, map, points_frame)
        solver.one_round(assoc)
        camera.updatePoseICP(solver.dx)
    
    print(f"T_abs:\n {np.round(camera.absolutePose(), decimals=2)}")
    camera.updateRelative()
    print(f"T_rel:\n {np.round(camera.relativePose(), decimals=2)}")


def countOut(map):
    out = 0 
    for elem in map:
        id, point = elem 
        if(point[2] <= 0):
            out+=1
    return out

def updateMap(map, point_prev_frame, point_curr_frame, R, t, assoc, camera, solver):
    #map[i] = (ID, (x, y, z))
    #point_prev_frame[i] = (ID, (x, y))
    #point_curr_frame[i] = (ID, (x, y))
    #assoc[i] = (ID, best_ID)

    id_in_map = [item[0] for item in map]
    id_curr_frame = [item[0] for item in point_curr_frame]
 
    #ID not in map
    missing = [item for item in id_curr_frame if item not in set(id_in_map)]
    
    points_prev = []
    points_curr = []
    for elem in missing:
        point_prev = u.getPoint(point_prev_frame, elem)
        point_curr = u.getPoint(point_curr_frame, elem)

        if(point_prev is not None and point_curr is not None):
            points_prev.append(point_prev)
            points_curr.append(point_curr)
    
    #ID to assigned for the new points of the map
    missing_assoc = [(id, best) for id, best in assoc if id in missing]

    #New triangulated points w.r.t. prev frame
    missing_map = u.triangulate(R, t, points_prev, points_curr, K, missing_assoc)

    test_picp(camera, solver, missing_map, point_curr_frame, missing_assoc)
    test_proj(missing_map, point_curr_frame, camera)

    for elem in missing_assoc:
        assoc.append(elem)

    #Update the map and the assoc
    for elem in missing_map:
        id, point = elem 
        point = np.array(point, dtype=np.float32)
        map.append((id, point))

    
    
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

    # Triangulate points w.r.t. frame 0
    map = u.triangulate(R, t, p0, p1, K, assoc)

    #versus(map, world_info)

    print("Frame 0:")
    test_proj(map, points_frame0, camera)
    
    print("P-ICP")
    test_picp(camera, solver, map, points_frame1, assoc)

    print("Frame 1:")
    test_proj(map, points_frame1, camera)
    
    iter = 2
    for i in range(1, iter):
        path_frame_prev = u.generate_path(i)
        path_frame_curr = u.generate_path(i+1)

        data_fame_prev = u.extract_measurements(path_frame_prev)
        data_frame_curr = u.extract_measurements(path_frame_curr)

        _, _, points_prev, points_curr, assoc = u.data_association(data_fame_prev, data_frame_curr)

        print("P-ICP before update map")
        test_picp(camera, solver, map, points_curr, assoc)

        print(f"Frame {i+1}:")
        test_proj(map, points_curr, camera) 

        #------Update the map------
        
        T_prev = u.g2T(gt[i])   # frame i in world frame (w_T_i)
        T_curr = u.g2T(gt[i+1]) # frame i+1 in world frame (w_T_i+1)
        
        #i_T_i+1 = i_T_w @ w_T_i+1 = inv(w_T_i) @ w_T_i+1
        T_rel = np.linalg.inv(T_prev) @ T_curr
        T_align = u.alignWithCameraFrame(T_rel)

        R_curr, t_curr = u.T2m(T_align)
        
        updateMap(map=map, 
                  point_prev_frame=points_prev, 
                  point_curr_frame=points_curr,
                  R=R_curr,
                  t=t_curr,
                  assoc=assoc,
                  camera=camera,
                  solver=solver)
        
        #------Update the map------


        test_proj(map, points_curr, camera)

        
if __name__ == '__main__':
    main() 