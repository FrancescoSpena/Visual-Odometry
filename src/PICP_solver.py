import numpy as np 
import utils as u

class PICP():
    def __init__(self, camera):
        self.camera = camera 
        self.world_points = None 
        self.image_points = None
        self.kernel_threshold = 1000
        
        self.keep_outliers = False
        self.num_inliers = 0
        self.chi_inliers = 0
        self.chi_outliers = 0
        self.min_num_inliers = 0
    
    def initial_guess(self, camera, world_points, reference_image_points):
        'Set the map and image points in a ref values'
        self.world_points = world_points
        self.image_points = reference_image_points
        self.camera = camera
    
    def error_and_jacobian(self, world_point, reference_image_point):
        'Compute error and jacobian given the 3D point and 2D point'
        status = True
        world_point_h = np.append(world_point, 1)
        camera_point = self.camera.worldInCameraPose() @ world_point_h
        camera_point = camera_point[:3] / camera_point[3]
        K = self.camera.cameraMatrix()
        predicted_image_point, is_valid = u.project_point(camera_point,
                                                          K)
        
        if not is_valid:
            status = False
            return None, None, status 
        
        error = predicted_image_point - reference_image_point

        world_point_h = np.append(world_point, 1)
        point_in_camera = self.camera.worldInCameraPose() @ world_point_h
        point_in_camera = point_in_camera[:3] / point_in_camera[3]
        
        J_r = np.zeros((3,6))
        J_r[:3, :3] = np.eye(3)
        J_r[:3, 3:] = u.skew(-point_in_camera)

        phom = K @ point_in_camera
        iz = 1.0 / phom[2]
        iz2 = iz*iz 

        J_p = np.array([
            [iz, 0, -phom[0]*iz2],
            [0, iz, -phom[1]*iz2]
        ])

        J = J_p @ K @ J_r

        return error, J, status
    
    def linearize(self, assoc):
        'Linearize the system and return H and b'
        H = np.zeros((6,6))
        b = np.zeros(6)

        for idx_frame1, idx_frame2 in assoc: 
            world_point = u.get_point(self.world_points,idx_frame2)
            image_point = u.get_point(self.image_points,idx_frame1)
            
            if world_point is None or image_point is None:
                continue

            error, J, status = self.error_and_jacobian(world_point, image_point)

            if status == False:
                continue

            chi = np.dot(error,error)
            lam = 1 
            is_inlier = True
            if(chi > self.kernel_threshold):
                lam = np.sqrt(self.kernel_threshold/chi)
                is_inlier = False 
                self.chi_outliers += chi 
            else: 
                self.chi_inliers += chi 
                self.num_inliers += 1
            
            if(is_inlier or self.keep_outliers):
                H += np.transpose(J) @ (J * lam)
                b += np.transpose(J) @ (error * lam)

        return H, b

    def solve(self, H, b): 
        'Solve a LS problem Ax = b'
        try: 
            dx = np.linalg.solve(H, -b)
        except:
            dx = np.zeros_like(b)
        return dx

    def one_round(self, assoc):
        'Compute dx'
        H, b = self.linearize(assoc)
        if(self.num_inliers < self.min_num_inliers):
            return
        self.dx = self.solve(H, b)
    
    def map(self):
        return self.world_points
    
    def points_2d(self):
        return self.image_points
    
    def set_map(self, points3d):
        self.world_points = points3d
    
    def set_image_points(self, image_points):
        self.image_points = image_points