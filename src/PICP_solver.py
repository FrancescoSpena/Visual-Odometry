import numpy as np 
import utils as u

class PICP():
    def __init__(self, camera):
        self.camera = camera 
        self.world_points = None 
        self.image_points = None
        self.kernel_threshold = 10000
    
    def initial_guess(self, world_points, reference_image_points):
        'Set the map and image points in a ref values'
        self.world_points = world_points
        self.image_points = reference_image_points
    
    def error_and_jacobian(self, world_point, reference_image_point):
        'Compute error and jacobian given the 3D point and 2D point'
        status = True
        K = self.camera.K
        predicted_image_point, is_true = u.project_point(world_point,
                                                         K)
        
        if(is_true == False):
            print("No good proj")
            status = False
        
        error = predicted_image_point - reference_image_point

        J_r = np.zeros((3,6))
        J_r[:3, :3] = np.eye(3)
        J_r[:3, 3:] = u.skew(world_point)

        phom = K @ world_point
        iz = 1.0 / phom[2]
        iz2 = iz * iz 

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
            
            #Non è detto che il punto 3D è visto dal frame corrente
            if world_point is None or image_point is None:
                continue

            error, J, status = self.error_and_jacobian(world_point, image_point)

            if status == False:
                continue

            chi = np.dot(error,error)
            lambda_factor = 1.0 

            if chi > self.kernel_threshold: 
                lambda_factor = np.sqrt(self.kernel_threshold / chi)

            H += J.T @ J * lambda_factor
            b += J.T @ error * lambda_factor

        return H, b

    def solve(self, H, b): 
        'Solve a LS problem Ax = b'
        try: 
            dx = np.linalg.solve(H, -b)
        except:
            dx = np.zeros_like(b)
        return dx

    def one_round(self, assoc):
        'Update the pose of the camera'
        H, b = self.linearize(assoc)
        dx = self.solve(H, b)
        self.camera.update_pose(dx)
    
    def map(self):
        return self.world_points
    
    def points_2d(self):
        return self.image_points
    
    def set_map(self, points3d):
        self.world_points = points3d
    
    def set_image_points(self, image_points):
        self.image_points = image_points