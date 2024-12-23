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
        
        #Absolute matrix (from 0 to idx+1)
        self.R = np.eye(3)
        self.t = np.zeros((3,1))
        
        #Relative matrix (from idx to idx+1)
        self.R_rel = np.eye(3)
        self.t_rel = np.zeros((3,1))
        
        self.status = True

    def init(self):
        path0 = u.generate_path(0)
        path1 = u.generate_path(1)

        data_frame_0 = u.extract_measurements(path0)
        data_frame_1 = u.extract_measurements(path1)

        #points0: frame 0, points1: frame1 
        points0, points1, _ = u.data_association(data_frame_0, 
                                              data_frame_1)
        
        #Pose from 0 to 1
        self.R, self.t = u.compute_pose(points0,
                                        points1,
                                        self.K)
        
        #3D points of the frame 0
        self.points3d_prev = u.triangulate(self.R,
                                      self.t,
                                      points0,
                                      points1,
                                      self.K)
        
        self.R_rel = self.R 
        self.t_rel = self.t
        
        if(np.linalg.det(self.R) != 1 or np.linalg.norm(self.t) == 0):
            self.status = False
        
        return self.R, self.t, self.status
    
    def run(self, idx):
        'Update pose in the frame idx+1'
        pass


        