import utils as u 
import data_manipulation as data
import numpy as np
import Camera as cam
import PICP_solver as s
import argparse
import testing as test
import visual_odometry as vo

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

#Camera and solver
camera = cam.Camera(K)
solver = s.PICP(camera=camera)

#List for accumulate the pose and information about the point
gt_pose = []
est_pose = []
pose_for_track = []
points_track = {}

def picp(map, points_curr, camera, assoc_3d, i):
    """
    Apply P-ICP to align the camera with the curr frame

    Args:
        map (list): (id, point, app) the triangulated map
        point_curr (list): (id, point) the measurements of the curr frame (without app)
        camera (obj.Camera)
        assoc_3d (list): (id, best) the association between the points in the map and the curr measures
        i (int): number of the frame
    Return:
        None

    """

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


def picp_app(map, points_curr, camera, assoc_3d, i):
    """
    Apply P-ICP to align the camera with the curr frame

    Args:
        map (list): (id, point, app) the triangulated map
        point_curr (list): (id, point) the measurements of the curr frame (without app)
        camera (obj.Camera)
        assoc_3d (list): (id, best) the association between the points in the map and the curr measures
        i (int): number of the frame
    Return:
        None

    """

    map_no_app = [(id, point) for id, point, _ in map]

    #Apply P-ICP to align the camera with the curr frame
    iter_picp = args.picp
    for _ in range(iter_picp):
        solver.initial_guess(camera, map_no_app, points_curr)
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
def process_data_for_frame(i, map, data_frame_prev, data_frame_curr):
    points_prev, points_curr = data.getMeasurementsFromDataFrame(data_frame_prev, data_frame_curr)

    T_i = camera.absolutePose().copy()

    T_rel_est = camera.relativePose().copy()
    R, t = u.T2m(T_rel_est)

    #------Retriangulation------
    map = vo.retriangulation_n_views(map=map, 
                                  est_pose=pose_for_track, 
                                  track=points_track, 
                                  measurements_curr=points_curr)
    #------Retriangulation------

    #------Update map------
    map = vo.updateMap(map, points_prev, points_curr, R, t, T_i)
    #------Update map------

    #------PICP------
    assoc_3d = data.association3d(map, points_curr, camera)
    picp(map, points_curr, camera, assoc_3d, i)
    #------PICP------


    for id, point in points_curr:
        vo.add_point_to_frame(points_track=points_track, frame_id=i+1, point_id=id, point=point)


def process_data_for_frame_app(i, map, data_frame_prev, data_frame_curr):
    points_prev_app, points_curr_app = data.getMeasurementsFromDataFrameApp(data_frame_prev, data_frame_curr)

    points_curr = [(id, point) for id, point, _ in points_curr_app]
    
    T_i = camera.absolutePose().copy()

    T_rel_est = camera.relativePose().copy()
    R, t = u.T2m(T_rel_est)

    #------Retriangulation------
    map = vo.retriangulation_n_views_app(map=map, 
                                  est_pose=pose_for_track, 
                                  track=points_track, 
                                  measurements_curr=points_curr)
    #------Retriangulation------

    #------Update map------
    assoc_for_update = vo.data_association_frame(points_prev_app, points_curr_app)
    map = vo.updateMapApp(map, points_prev_app, points_curr_app, R, t, T_i, assoc_for_update)
    #------Update map------

    #------PICP------
    assoc_3d = data.association3d_with_similarity(map, points_curr_app, camera)
    picp_app(map, points_curr, camera, assoc_3d, i)
    #------PICP------

    for id, point in points_curr:
        vo.add_point_to_frame(points_track=points_track, frame_id=i+1, point_id=id, point=point)
    


def process_frame(i, map):
    print(f"From frame {i} to {i+1}")
    path_frame_prev = u.generate_path(i)
    path_frame_curr = u.generate_path(i+1)

    data_frame_prev = u.extract_measurements(path_frame_prev)
    data_frame_curr = u.extract_measurements(path_frame_curr)

    process_data_for_frame_app(i, map, data_frame_prev=data_frame_prev, data_frame_curr=data_frame_curr)

    

def main():
    print("From frame 0 to 1")
    path_frame0 = u.generate_path(0)
    path_frame1 = u.generate_path(1)

    data_frame0 = u.extract_measurements(path_frame0)
    data_frame1 = u.extract_measurements(path_frame1)

    p0, p1, points_frame0_app, points_frame1_app, assoc = data.data_association_with_similarity(data_frame0, data_frame1)

    points_frame0 = [(id, point) for id, point, _ in points_frame0_app]
    points_frame1 = [(id, point) for id, point, _ in points_frame1_app]
    app = [(app) for _, _, app in points_frame1_app]


    for id, point in points_frame0:
        vo.add_point_to_frame(points_track=points_track, frame_id=0, point_id=id, point=point)
    
    for id, point in points_frame1:
        vo.add_point_to_frame(points_track=points_track, frame_id=1, point_id=id, point=point)


    #----------Complete VO-----------
    #Good rotation and translation is consistent to the movement (forward)
    
    R, t = data.compute_pose(points_frame0, points_frame1, K=K)
    T1_est = u.m2T(R,t)

    #----------Complete VO-----------

    #----------GT-----------
    T0_gt = u.g2T(gt[0])  # frame 0 in world frame (w_T_0)
    T1_gt = u.g2T(gt[1])  # frame 1 in world frame (w_T_1)

    T_rel_gt = u.relativeMotion(T0_gt, T1_gt)
    T_rel_gt = np.round(T_rel_gt, 2)

    #----------GT-----------
    gt_pose.append(T0_gt)
    gt_pose.append(T1_gt)

    # Triangulate points w.r.t. frame 0
    map = data.triangulateWithApp(R, t, p0, p1, K, assoc, app_curr_frame=app)
    camera.setCameraPose(T1_est)

    #Pose align w.r.t. world frame
    est_pose.append(T0_gt)
    est_pose.append(u.alignWithWorldFrame(T1_est))

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


    ratio = test.evaluate(est_pose, gt_pose)
    scale_est = test.scale_est_poses(est_pose=est_pose, scale_ratio=ratio)

    if(args.plot):
        test.plot(gt_pose, scale_est)





if __name__ == '__main__':
    main()

