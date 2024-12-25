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
        points0, points1, assoc = u.data_association(data_frame_0, 
                                              data_frame_1)
        
        #Pose from 0 to 1
        self.R, self.t = u.compute_pose(points0,
                                        points1,
                                        self.K)
        
        #3D points of the frame 0
        self.points_3d_curr = u.triangulate(self.R,
                                      self.t,
                                      points0,
                                      points1,
                                      self.K,
                                      assoc)
        
        self.R_rel = self.R 
        self.t_rel = self.t
        
        if(np.linalg.det(self.R) != 1 or np.linalg.norm(self.t) == 0):
            self.status = False
        
        return self.R, self.t, self.status
    
    def run(self, idx):
        'Update pose in the frame idx+1'
        path_curr_frame = u.generate_path(idx)
        path_next_frame = u.generate_path(idx+1)

        data_curr = u.extract_measurements(path_curr_frame)
        data_next = u.extract_measurements(path_next_frame)

        points_curr, points_next, assoc = u.data_association(data_curr,
                                                             data_next)
        
        world_curr_frame = u.triangulate(self.R,
                                         self.t,
                                         points_curr,
                                         points_next,
                                         self.K,
                                         assoc)
        

        world_next_frame = []

        for curr, next in assoc: 
            point = u.get_point(world_curr_frame,curr)
            if point is not None:
                new_tuple = (next, u.get_point(world_curr_frame,curr))
                world_next_frame.append(new_tuple)
            else:
                print("Not exist.")

        for curr, next in assoc:
            print(f"Point ID {curr} -> Point ID {next}")
            print(f"idx: {curr}, 3d point curr: {u.get_point(world_curr_frame,curr)}")
            print(f"idx: {next}, 3d point next: {u.get_point(world_next_frame,next)}")
            print("=======")

        # #Linearize the sys
        # H, b = u.linearize(assoc,
        #                    world_next_frame,
        #                    points_next,
        #                    self.K)
        
        # #Compute delta_pose (solve LS problem)
        # dx = u.solve(H, b)

        # #Update pose (boxplus operator)
        # T_curr = u.m2T(self.R, self.t)      #pose from 0 to idx 
        # T = u.v2T(dx) @ T_curr              #apply boxplus operator --> update pose (from 0 to idx+1)
        # T_rel = np.linalg.inv(T_curr) @ T 

        # self.R_rel, self.t_rel = u.T2m(T_rel)
        # self.R, self.t = u.T2m(T)

        return self.R_rel, self.t_rel
            
        


        