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
    
    def initial_guess(self, camera, world_points, point_curr_frame):
        'Set the map and image points in a ref values'
        self.world_points = world_points
        self.image_points = point_curr_frame
        self.camera = camera
    
    def error_and_jacobian(self, world_point, reference_image_point):
        'Compute error and jacobian given the 3D point and 2D point'
        status = True

        K = self.camera.cameraMatrix()

        fx = K[0, 0]
        fy = K[1, 1]
        
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

        x_cam = camera_point[0]
        y_cam = camera_point[1]
        z_cam = camera_point[2]
        
        J = np.zeros((2,6))

        first_col = np.array([fx / z_cam, 0])
        second_col = np.array([0, fy / z_cam])
        third_col = np.array([-fx * (x_cam / (z_cam**2)), -fy * (y_cam / (z_cam**2))])
        four_col = np.array([fx * ((x_cam * y_cam) / (z_cam**2)), fy * (1 + ((y_cam)**2) / (z_cam**2))])
        five_col = np.array([-fx * (1 + ((x_cam)**2) / (z_cam**2)), -fy * ((y_cam * x_cam) / (z_cam**2))])
        six_col = np.array([fx * (y_cam / z_cam), -fy * (x_cam / z_cam)])


        J = np.column_stack((first_col, 
                             second_col, 
                             third_col, 
                             four_col, 
                             five_col, 
                             six_col))
        
        return error, J, status
    
    def linearize(self, assoc):
        'Linearize the system and return H and b'
        H = np.zeros((6,6))
        b = np.zeros((6,1))

        for _, best_id in assoc: 
            world_point = u.get_point(self.world_points,best_id)
            image_point = u.get_point(self.image_points,best_id)

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