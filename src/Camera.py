import numpy as np
import utils as u

class Camera():
    def __init__(self, K, z_near=0, z_far=5, width=640, height=480):
        self.prev_T_abs = np.eye(4)
        #Is a transformation from the world to the camera (express the world in camera frame)
        # w_T_c
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
        'Return the relative pose'
        return self.T_rel
    
    def updateRelative(self):
        self.T_rel = np.linalg.inv(self.prev_T_abs) @ self.T_abs
    
    def updatePrev(self):
        self.prev_T_abs = self.T_abs

    def updatePoseICP(self, dx):
        'Update pose (from 0 to frame i+1 and save T_rel)'
        self.T_abs = u.v2T(dx) @ self.T_abs

    def setCameraPose(self, pose):
        'Update the absolute pose with the pose (T_abs=pose)'
        self.T_rel = pose
        self.T_abs = pose
        self.prev_T_abs = pose
    
    def cameraMatrix(self):
        'Return the camera matrix'
        return self.K
    
    def proj(self, u):
        'prospective projection'
        return np.array([u[0]/u[2], u[1]/u[2]])

    def project_point(self, world_point, width=640, height=480, z_near=0, z_far=5): 
        'Project on the image the world_point=world_point'
        'image_point = proj(K T_cam^-1 * world_point)'
        'proj(u) = (ux / uz     uy / uz)'

        #world in camera
        w_T_c = self.absolutePose()
        world_point = np.append(world_point, 1)
        
        # (4 x 1)
        p_cam = w_T_c @ world_point

        z = p_cam[2]

        if(z <= z_near):
            # print("Point out of camera view")
            # print(f"z: {z}")
            # print("-------------------")
            return None, False
        
        # (3 x 1)
        image_point = self.cameraMatrix() @ p_cam[:3]
        
        # (2 x 1)
        image_point = self.proj(image_point)

        x = image_point[0]
        y = image_point[1]

        if(x < 0 or x > width-1):
            # print("Point out of image size")
            # print(f"x: {x}")
            # print("-------------------")
            return None, False 
        
        if(y < 0 or y > height-1):
            # print("Point out of image size")
            # print(f"y: {y}")
            # print("-------------------")
            return None, False 

        # print("VALID")
        # print("================")
        return image_point, True
    
    def unprojectPixelToRay(self, pixel):
        pixel_h = np.array([pixel[0], pixel[1], 1])
        ray_direction = np.linalg.inv(self.cameraMatrix()) @ pixel_h
        ray_direction /= np.linalg.norm(ray_direction)
        return ray_direction