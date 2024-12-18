import utils as u
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat', traj_path='../data/trajectoy.dat', optim=50):
        self.camera_info = u.extract_camera_data(camera_path)
        self.K = self.camera_info['camera_matrix']
        self.width = self.camera_info['width']
        self.height = self.camera_info['height']
        self.z_near = self.camera_info['z_near']
        self.z_far = self.camera_info['z_far']
        
        self.traj = u.read_traj(traj_path)
        
        self.poses_camera = []
        self.R = np.eye(3)
        self.t = np.zeros((3,1))
        
        self.idx = 0
        self.all_2d_points = []
        self.all_3d_points = []
        self.optim = optim

    def run(self, idx):
        '''Return the transformation between the frame idx and idx+1'''
        #Initialization

        #Extract data from measurements
        first_data = u.extract_measurements(u.generate_path(idx))
        second_data = u.extract_measurements(u.generate_path(idx+1))
            
        #data association
        assoc1, assoc2 = u.data_association(first_data,
                                   second_data)

        points1, points2 = u.extract_points(first_data,assoc1), u.extract_points(second_data,assoc2)
        
        res = u.compute_pose(points1,
                             points2,
                             self.K,
                             self.z_near,
                             self.z_far)
        if res is not None:
            self.R, self.t = res
            
        self.poses_camera.append((self.R,self.t))
        return u.m2T(self.R, self.t)
        