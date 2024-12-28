import numpy as np
import utils as u

class Camera():
    def __init__(self, K, z_near=0, z_far=5, width=640, height=480):
        #Absolute pose
        self.T = np.eye(4)
        #Relative pose
        self.T_rel = np.eye(4)
        
        self.K = K
        self.z_near = z_near
        self.z_far = z_far
        self.width = width
        self.height = height
    
    def update_pose(self, dx):
        'Update the pose of the camera given the delta (boxplus operator)'
        T_curr = self.T
        self.T = u.v2T(dx) @ T_curr
        self.T_rel = np.linalg.inv(T_curr) @ self.T

    def update_absolute(self, T):
        self.T = T
    
    def update_relative(self, T_rel):
        self.T_rel = T_rel
    
    def absolute_pose(self):
        return self.T

    def relative_pose(self):
        return self.T_rel

