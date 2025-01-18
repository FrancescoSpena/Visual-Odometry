import numpy as np
import utils as u

class Camera():
    def __init__(self, K, z_near=0, z_far=5, width=640, height=480):
        self.world_in_camera_pose = np.eye(4)
        self.K = K
        
        self.z_near = z_near
        self.z_far = z_far
        self.width = width
        self.height = height
    
    def worldInCameraPose(self):
        return self.world_in_camera_pose
    
    def setWorldInCameraPose(self, pose):
        self.world_in_camera_pose = pose
    
    def cameraMatrix(self):
        return self.K
    
    def updatePose(self, dx):
        self.setWorldInCameraPose(u.v2T(dx) @ self.worldInCameraPose())

