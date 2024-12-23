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

        points1, points2, _ = u.data_association(data_frame_0, 
                                              data_frame_1)
        
        self.R, self.t = u.compute_pose(points1,
                                        points2,
                                        self.K)
        
        self.R_rel = self.R 
        self.t_rel = self.t
        
        if(np.linalg.det(self.R) != 1 or np.linalg.norm(self.t) == 0):
            self.status = False
        
        return self.R, self.t, self.status
    
    def run(self, idx):
        'Return the transformation from frame idx and idx+1'
        
        #Path for the file of the frame idx and idx+1
        path_curr_frame = u.generate_path(idx)
        path_next_frame = u.generate_path(idx+1)
        
        #Dict for frame idx and idx+1
        data_curr_frame = u.extract_measurements(path_curr_frame)
        data_next_frame = u.extract_measurements(path_next_frame)

        #Data association -> points frame idx, points frame idx+1, assoc=(idx, best_idx)
        points_curr, points_next, assoc = u.data_association(data_curr_frame,
                                                             data_next_frame)
        
        #World points of the frame idx
        world_points = u.triangulate(self.R_rel,
                                     self.t_rel,
                                     points_curr,
                                     points_next,
                                     self.K)
        
        
        #World points of the frame idx+1
        world_points = (self.R_rel @ world_points.T).T + self.t_rel.T
        
        #Linearize the system
        H, b, status = u.linearize(assoc,
                              world_points,
                              points_curr,
                              self.K)

        if status == False: 
            self.status = False
        
        #Compute the delta of the pose (solve LS problem)
        self.dx = u.solve(H, b)
        
        #Update the pose (boxplus operator)
        T_curr = u.m2T(self.R, self.t)          #frame 0 to idx
        T = u.v2T(self.dx) @ T_curr             #frame 0 to idx+1
        T_rel = np.linalg.inv(T_curr) @ T       #frame idx to idx+1

        self.R_rel, self.t_rel = u.T2m(T_rel)
        self.R, self.t = u.T2m(T)
        
        return self.R_rel, self.t_rel, self.status