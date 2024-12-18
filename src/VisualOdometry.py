import utils as u
import numpy as np
import cv2

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

    # Return a transformation between frame i and i+1
    def run(self, idx):
        pass