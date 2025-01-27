import numpy as np
import utils as u

class Camera():
    def __init__(self, K, z_near=0, z_far=5, width=640, height=480):
        self.T_abs = np.eye(4)
        self.T_rel = np.eye(4)
        
        self.K = K
        
        self.z_near = z_near
        self.z_far = z_far
        self.width = width
        self.height = height
        self.tolerance = 1e-2
    
    def absolutePose(self):
        'Return the absolute pose (from frame 0 to frame i+1)'
        return self.T_abs

    def relativePose(self):
        'Return the relative pose (from frame i to frame i+1)'
        return self.T_rel
    
    def updatePose(self, T_rel):
        'Update pose (from 0 to frame i+1 and save T_rel)'
        self.T_rel = T_rel
        self.T_abs = self.T_abs @ self.T_rel
    
    def setCameraPose(self, pose):
        'Update the absolute pose with the pose (T_abs=pose)'
        self.T_rel = pose
        self.T_abs = pose
    
    def cameraMatrix(self):
        'Return the camera matrix'
        return self.K
    
    def project_point(self, world_point, width=640, height=480, z_near=0, z_far=5): 
        'Project on the image the world_point=world_point'
        image_point = np.zeros((2,))
        world_point_h = np.append(world_point, 1)
        camera_point = self.absolutePose() @ world_point_h
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



