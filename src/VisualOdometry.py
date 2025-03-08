import utils as u
import numpy as np
import PICP_solver as solver
import Camera as camera
import matplotlib.pyplot as plt
from collections import Counter

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

        _, _, points0_frame, points1_frame, assoc = u.data_association(data_frame_0, data_frame_1)

        #Pose from 0 to 1
        R, t = u.compute_pose(points0_frame,
                              points1_frame,
                              self.cam.K)
        
        T = u.m2T(R,t)
        self.cam.setCameraPose(T)

        #Check
        if(not np.isclose(np.linalg.det(R), 1, atol=1e-6) or np.linalg.norm(t) == 0):
            print(f"det(R): {np.linalg.det(R)}")
            print(f"norm(t): {np.linalg.norm(t)}")
            self.status = False
        
        return self.status
              