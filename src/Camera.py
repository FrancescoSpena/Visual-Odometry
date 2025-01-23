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
        self.tolerance = 1e-2
    
    def worldInCameraPose(self):
        return self.world_in_camera_pose
    
    def setWorldInCameraPose(self, pose):
        self.world_in_camera_pose = pose
    
    def cameraMatrix(self):
        return self.K
    
    def updatePose(self, dx):
        self.setWorldInCameraPose(u.v2T(dx) @ self.worldInCameraPose())
    
    def project_point(self, world_point, width=640, height=480, z_near=0, z_far=5): 
        image_point = np.zeros((2,))
        world_point_h = np.append(world_point, 1)
        camera_point = self.worldInCameraPose() @ world_point_h
        camera_point = camera_point[:3] / camera_point[3]

        if camera_point[2] <= z_near or camera_point[2] > z_far + self.tolerance:
            return image_point, False
        
        projected_point = self.cameraMatrix() @ camera_point
        image_point = projected_point[:2] * (1. / projected_point[2])

        if image_point[0] < 0 or image_point[0] > width-1: 
            return image_point, False 
        if image_point[1] < 0 or image_point[1] > height-1:
            return image_point, False
        
        return image_point, True



