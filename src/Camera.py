import numpy as np
import utils as u

class Camera():
    def __init__(self, K, z_near=0, z_far=5, width=640, height=480):
        '''
        Small explanation: 
        T_abs      are the homogeneous transformation that express the world in camera frame i+1. 
        T_rel      are the homogeneous transformation that express the relative motion between frame i and i+1. 
        prev_T_abs are the homogeneous transformation that express the world in camera frame i. 
        '''
        
        self.prev_T_abs = np.eye(4)
        self.T_abs = np.eye(4)
        self.T_rel = np.eye(4)
        
        self.K = K
        
        self.z_near = z_near
        self.z_far = z_far
        self.width = width
        self.height = height
    
    def absolutePose(self):
        'Return the absolute pose (from frame 0 to frame i+1)'
        return self.T_abs
    
    def relativePose(self):
        'Return the relative pose'
        return self.T_rel
    
    def updateRelative(self, T_abs_new):
        'Compute the relative pose given a new absolute pose'
        inv = np.linalg.inv(self.T_abs)
        self.T_rel = inv @ T_abs_new
        return self.T_rel
        
    def updatePoseICP(self, dx):
        'Update absolute pose'
        self.T_abs = u.v2T(dx) @ self.T_abs

    def setCameraPose(self, pose):
        'Update the absolute pose with the pose (T_abs=pose)'
        self.T_abs = pose
        self.T_rel = pose
        self.prev_T_abs = pose
    
    def cameraMatrix(self):
        'Return the camera matrix'
        return self.K
    
    def proj(self, u):
        'Prospective projection'
        return np.array([u[0]/u[2], u[1]/u[2]])

    def project_point(self, world_point, width=640, height=480, z_near=0, z_far=5): 
        'Project on the image the world_point=world_point'
        'image_point = proj(K T_cam^-1 * world_point)'
        'proj(u) = (ux / uz     uy / uz)'

        # World in camera frame
        c_T_w = self.T_abs
        # Homogeneous coordinate
        world_point = np.append(world_point, 1)
        
        # (4 x 1)  p_i = i_T_0 @ p_0
        p_cam = c_T_w @ world_point

        z = p_cam[2]

        if(z <= z_near or z > z_far):
            # print(f"[Camera][project_point]Point out of camera view, z: {z}")
            return None, False
        
        # (3 x 1)
        image_point = self.cameraMatrix() @ p_cam[:3]
        
        # (2 x 1)
        image_point = self.proj(image_point)

        x = image_point[0]
        y = image_point[1]

        if(x < 0 or x > width-1):
            # print(f"[Camera][project_point]Point out of image size, x: {x}")
            return None, False 
        
        if(y < 0 or y > height-1):
            # print(f"[Camera][project_point]Point out of image size, y: {y}")
            return None, False 

        # print("VALID")
        # print("================")
        return image_point, True

