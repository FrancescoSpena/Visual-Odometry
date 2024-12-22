import utils as u
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat'):
        self.camera_info = u.extract_camera_data(camera_path)
        self.K = self.camera_info['camera_matrix']
        self.width = self.camera_info['width']
        self.height = self.camera_info['height']
        self.z_near = self.camera_info['z_near']
        self.z_far = self.camera_info['z_far']
        
        self.R = np.eye(3)
        self.t = np.zeros((3,1))
        self.points_3d = None
        self.status = True
        self.max_iter = 100

    def init(self):
        path0 = u.generate_path(0)
        path1 = u.generate_path(1)

        data_frame_0 = u.extract_measurements(path0)
        data_frame_1 = u.extract_measurements(path1)

        points1, points2 = u.data_association(data_frame_0, 
                                              data_frame_1)
        
        self.R, self.t = u.compute_pose(points1,
                                        points2,
                                        self.K)
        
        self.points_3d = u.triangulate(self.R, 
                                       self.t,
                                       points1,
                                       points2,
                                       self.K)
        
        if(np.linalg.det(self.R) != 1 or np.linalg.norm(self.t) == 0):
            self.status = False
        
        return self.R, self.t, self.points_3d, self.status
    
    def run(self, idx):
        path_curr_frame = u.generate_path(idx)
        path_next_frame = u.generate_path(idx+1)
        
        data_curr_frame = u.extract_measurements(path_curr_frame)
        data_next_frame = u.extract_measurements(path_next_frame)

        for i in range(self.max_iter):
            transformed_data = u.transform_point_in_dict(data_curr_frame,
                                                         self.R,
                                                         self.t)

            original_frame_points, transformed_frame_points = u.data_association(transformed_data,
                                                                    data_next_frame)

            pass