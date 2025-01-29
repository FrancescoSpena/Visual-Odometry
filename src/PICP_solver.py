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
    
    def initial_guess(self, camera, world_points, point_prev_frame):
        'Set the map and image points in a ref values'
        self.world_points = world_points
        self.image_points = point_prev_frame
        self.camera = camera
    
    def error_and_jacobian(self, world_point, reference_image_point):
        'Compute error and jacobian given the 3D point and 2D point'

        K = self.camera.cameraMatrix()

        predicted_image_point, is_valid = self.camera.project_point(world_point)
        
        if (is_valid == False):
            return None, None, False 
        
        # (2 x 1)
        error = np.array([
            [predicted_image_point[0] - reference_image_point[0]],
            [predicted_image_point[1] - reference_image_point[1]]
        ])

        # (x_cam, y_cam, z_cam)
        camera_point = u.w2C(world_point, self.camera.absolutePose())

        Jr = np.zeros((3,6))
        Jr[:3, :3] = np.eye(3)
        Jr[:3, 3:6] = u.skew(-camera_point)

        phom = K @ camera_point
        iz = 1./phom[2]
        iz2 = iz ** 2 

        Jp = np.zeros((2,3))
        Jp[0, :] = [iz, 0, -phom[0] * iz2]
        Jp[1, :] = [0, iz, -phom[1] * iz2]

        # (2, 6) = (2 x 3) * (3 x 3) * (3 x 6)
        J = Jp @ K @ Jr
      
        return error, J, True
    
    def linearize(self, assoc):
        'Linearize the system and return H and b'
        H = np.zeros((6,6))
        b = np.zeros((6,1))

        for i, (id, best_id) in enumerate(assoc):
            curr_idx = best_id
            
            world_point = u.getPoint3D(self.world_points,curr_idx)
            image_point = self.image_points[i]

            if world_point is None or image_point is None:
                continue

            error, J, status = self.error_and_jacobian(world_point, image_point)

            if status == False:
                continue

            chi = np.dot(error.T,error)
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
                # (6 x 2) * (2 x 6) = (6 x 6)
                H += J.T @ J * lam
                # (6 x 2) * (2 x 1) = (6 x 1)
                b += J.T @ error * lam

        return H, b

    def solve(self, H, b): 
        'Solve a LS problem H*delta_x = -b'
        try:
            return np.linalg.solve(H, -b).reshape(-1, 1)
        except:
            print("Singular matrix")
            print(f"H:\n {H}")
            print(f"b:\n {b}")
            print("------------------")
            return np.zeros((1,6))

    def one_round(self, assoc):
        'Compute dx'
        H, b = self.linearize(assoc)
        if(self.num_inliers < self.min_num_inliers):
            return
        # (1 x 6)
        self.dx = self.solve(H, b).T
    
    def getMap(self):
        return self.world_points
    
    def points_2d(self):
        return self.image_points
    
    def set_map(self, points3d):
        self.world_points = points3d
    
    def set_image_points(self, image_points):
        self.image_points = image_points