import numpy as np 
import utils as u

class PICP():
    def __init__(self, camera):
        self.camera = camera 
        self.world_points = None 
        self.image_points = None
        self.kernel_threshold = 10000
        
        self.keep_outliers = False
        self.num_inliers = 0
        self.chi_inliers = 0
        self.chi_outliers = 0
        self.min_num_inliers = 0
        self.damping = 0.1
    
    def initial_guess(self, camera, world_points, reference_image_points):
        'Set the map and image points in a ref values'
        self.world_points = world_points
        self.image_points = reference_image_points
        self.camera = camera
    
    def error_and_jacobian(self, world_point, reference_image_point):
        'Compute error and jacobian given the 3D point and 2D point'
        status = True

        K = self.camera.cameraMatrix()
        predicted_image_point, is_valid = self.camera.project_point(world_point)
        
        if (is_valid == False):
            return None, None, False 
        
        error = predicted_image_point - reference_image_point

        camera_point = u.w2C(world_point, self.camera.absolutePose())
        
        J_r = np.zeros((3,6))
        J_r[:3, :3] = np.eye(3)
        J_r[:, 3:6] = u.skew(-camera_point)

        phom = K @ camera_point
        
        iz = 1. / phom[2]
        iz2 = iz*iz 

        J_p = np.array([
            [iz, 0, -phom[0]*iz2],
            [0, iz, -phom[1]*iz2]
        ])

        # (2 x 3) @ (3 x 3) @ (3 x 6)
        # (2 x 3) @ (3 x 6)
        # (2 x 6)
        J = J_p @ K @ J_r


        return error, J, status
    
    def linearize(self, assoc):
        'Linearize the system and return H and b'
        H = np.zeros((6,6))
        b = np.zeros(6)

        for idx_frame1, idx_frame2 in assoc: 
            ref_idx = idx_frame1
            curr_idx = idx_frame2
            world_point = u.get_point(self.world_points,curr_idx)
            image_point = u.get_point(self.image_points,ref_idx)

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
                H += J.T @ J * lam
                b += J.T @ error * lam

        return H, b

    def solve(self, H, b): 
        'Solve a LS problem H*delta_x = -b'
        return np.linalg.solve(H, -b)

    def one_round(self, assoc):
        'Compute dx'
        H, b = self.linearize(assoc)
        H += np.eye(6) * self.damping
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