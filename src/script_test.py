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

def getPoint(point_frame0, target_id):
    'Return 2D given the id'
    for elem in point_frame0:
        id, point = elem 

        if(id == target_id):
            point = [float(x) for x in point]
            point = np.array(point)
            return point
    
    return None

def transform_point(p_cam, T):
    p_homog = np.append(p_cam, 1.0)
        #w_p = w_T_c @ c_p
    p_world_homog = T @ p_homog
    return p_world_homog[:3]
    
def versus(map, world):
    #w_T_c
    #T = u.alignWithWorldFrame(T_align)

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
    
def test_proj(map, point_frame0, camera):
    all_equal = True
    i = 0
    for elem in map:
        id, point = elem
        project_point, isvalid = camera.project_point(point)
        
        if isvalid:
            point_true = getPoint(point_frame0, id)
            if(point_true is not None):
                print(f"true point: {point_true}, proj: {project_point}")
                print("------")
        
        # if(i % 10 == 0):
        #     break
        i +=1
        
    return all_equal  
    
def test():
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

def test_picp(camera, solver, map, points, assoc):
    camera.updatePrev()
    for _ in range(5):
        solver.initial_guess(camera, map, points)
        solver.one_round(assoc)
        camera.updatePoseICP(solver.dx)
    
    print(f"T_abs:\n {np.round(camera.absolutePose(), decimals=3)}")
    camera.updateRelative()
    print(f"T_rel:\n {np.round(camera.relativePose(), decimals=3)}")

def main():
    camera = cam.Camera(K)
    solver = s.PICP(camera=camera)
    path_frame0 = u.generate_path(0)
    path_frame1 = u.generate_path(1)

    data_frame0 = u.extract_measurements(path_frame0)
    data_frame1 = u.extract_measurements(path_frame1)

    p0, p1, points_frame0, points_frame1, assoc = u.data_association(data_frame0, data_frame1)
    
    #----------Complete VO-----------
    #Good rotation and translation is consistent to the movement (forward)
    
    v = vo.VisualOdometry()
    status = v.init()
    print(f"Status: {status}")
    T = v.cam.absolutePose()
    R, t = u.T2m(T)

    #print(f"R:\n {R}, \nt:\n {t}")
    
    #----------Complete VO-----------

    #----------Internal VO-----------
    #With this the test=versus is scaled (value large but its normal)
    
    # R, t = u.compute_pose(p0, p1, K)

    # R = np.round(R)
    # t = np.round(t)

    # print(f"R:\n {R}, \nt:\n {t}")
    
    #----------Internal VO-----------

    #----------GT-----------
    # T0_gt = u.g2T(gt[0])  # frame 0 in world frame (w_T_0)
    # T1_gt = u.g2T(gt[1])  # frame 1 in world frame (w_T_1)

    # # Compute relative pose: 0_T_1 = 0_T_w @ w_T_1
    # T_rel = np.linalg.inv(T0_gt) @ T1_gt

    # # print(T_rel)
    # #T_align: transformation align with camera frame
    # #c_T_w
    # T_align = u.alignWithCameraFrame(T_rel)

    # R, t = u.T2m(T_align)

    # print(f"R:\n {R}, \nt:\n {t}")

    #----------GT-----------

    # Triangulate points w.r.t. camera frame
    map = u.triangulate(R, t, p0, p1, K, assoc)

    #versus(map, world_info)

    print("Frame 0:")
    test_proj(map, points_frame0, camera)
    
    print("P-ICP")
    test_picp(camera, solver, map, p1, assoc)

    print("Frame 1:")
    test_proj(map, points_frame1, camera)

    path_frame1 = u.generate_path(1)
    path_frame2 = u.generate_path(2)

    data_frame1 = u.extract_measurements(path_frame1)
    data_frame2 = u.extract_measurements(path_frame2)

    p1, p2, points_frame1, points_frame2, assoc = u.data_association(data_frame1, data_frame2)

    print("P-ICP")
    test_picp(camera, solver, map, p2, assoc)

    print("Frame 2:")
    test_proj(map, points_frame2, camera)




if __name__ == '__main__':
    main() 