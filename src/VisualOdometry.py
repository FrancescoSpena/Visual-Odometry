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
        
        self.gt = u.read_traj(traj_path)
        
        self.poses_camera = []
        self.R = np.eye(3)
        self.t = np.zeros((3,1))
        
        self.idx = 0
        self.all_2d_points = []
        self.all_3d_points = []
        self.optim = optim

    def run(self, idx):
        '''Return the transformation between the frame idx and idx+1'''
        self.idx = idx
        if(self.idx == 0):
            #Initialization

            #Extract data from measurements
            first_data = u.extract_measurements(u.generate_path(self.idx))
            second_data = u.extract_measurements(u.generate_path(self.idx+1))
            
            #data association
            assoc = u.data_association(first_data, second_data)

            points1 = np.array([
                (assoc_item['Image_X_First'], assoc_item['Image_Y_First'])
                for assoc_item in assoc
            ])

            points2 = np.array([
                (assoc_item['Image_X_Second'], assoc_item['Image_Y_Second'])
                for assoc_item in assoc
            ])

            if len(points1) < 5: 
                print("No sufficient point, identity")
                print("==================")
                self.R = np.eye(3)
                self.t = np.zeros((3,1))
            else:
                self.R, self.t = u.compute_pose(points1,
                                                points2,
                                                self.K)
        
            self.poses_camera.append((self.R, self.t))
            self.prev_data = second_data
            return u.m2T(self.R, self.t)
        else:
            curr_data = u.extract_measurements(u.generate_path(idx+1))
            assoc = u.data_association(self.prev_data,curr_data)

            prev_points = np.array([
                (assoc_item['Image_X_First'], assoc_item['Image_Y_First'])
                for assoc_item in assoc
            ])

            curr_points = np.array([
                (assoc_item['Image_X_Second'], assoc_item['Image_Y_Second'])
                for assoc_item in assoc
            ])

            if len(prev_points) < 5: 
                print("No sufficient point, last pose used")
                print("============")
            else: 
                self.R, self.t = u.compute_pose(prev_points, 
                                                curr_points, 
                                                self.K)
            
            self.poses_camera.append((self.R, self.t))
            self.prev_data = curr_data
            return u.m2T(self.R, self.t)
            