import utils as u
import numpy as np
import PICP_solver as solver
import Camera as camera
import cv2
import matplotlib.pyplot as plt

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat'):
        self.camera_info = u.extract_camera_data(camera_path)
        K = self.camera_info['camera_matrix']
        self.cam = camera.Camera(K)
        self.solver = solver.PICP(self.cam)
        self.status = True

    def init(self):
        path0 = u.generate_path(0)
        path1 = u.generate_path(1)

        data_frame_0 = u.extract_measurements(path0)
        data_frame_1 = u.extract_measurements(path1)

        other_info_frame_0 = u.extract_other_info(path0)
        other_info_frame_1 = u.extract_other_info(path1)

        self.prev_frame = data_frame_0
        
        #points0: frame 0, points1: frame1 --> (ID, (X,Y))
        points0, points1, assoc = u.data_association(data_frame_0, 
                                                     data_frame_1)
        
        p_0 = np.array([item[1] for item in points0])
        p_1 = np.array([item[1] for item in points1])

        gt_0 = np.array(other_info_frame_0['Ground_Truths'])
        gt_1 = np.array(other_info_frame_1['Ground_Truths'])

        gt_dist = np.linalg.norm(gt_1 - gt_0)

        #Pose from 0 to 1
        R, t = u.compute_pose(p_0,
                              p_1,
                              self.cam.K,
                              gt_dist)
        
        #3D points of the frame 0
        points_3d = u.triangulate(R,
                                  t,
                                  p_0,
                                  p_1,
                                  self.cam.K,
                                  assoc)
        
    
        self.cam.update_absolute(u.m2T(R, t))
        self.cam.update_relative(u.m2T(R, t))

        self.solver.set_map(points_3d)
        self.solver.set_image_points(points0)

        #Check
        if(np.linalg.det(R) != 1 or np.linalg.norm(t) == 0):
            self.status = False
        
        return self.status
    
    def run(self, idx):
        'Update pose'
        path_curr = u.generate_path(idx)
        data_curr = u.extract_measurements(path_curr)

        _, points_curr, assoc = u.data_association(self.prev_frame,data_curr)

        world_points = self.solver.map()
        point_prev = self.solver.points_2d()
        self.solver.initial_guess(world_points, point_prev)
        self.solver.one_round(assoc)
        self.solver.set_image_points(points_curr)
        self.prev_frame = data_curr