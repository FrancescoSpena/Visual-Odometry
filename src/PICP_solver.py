import numpy as np 
import utils as u
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix

class PICP():
    def __init__(self, camera):
        self.camera = camera 
        self.world_points = None 
        self.image_points = None
        self.damping = 0.0001
    
    def initial_guess(self, camera, world_points, point_curr_frame):
        'Set the map and image points in a ref values'
        self.world_points = world_points
        self.image_points = point_curr_frame
        self.camera = camera
    
    def error_and_jacobian(self, world_point, reference_image_point):
        'Compute error and jacobian given the 3D point and 2D point'

        K = self.camera.cameraMatrix()

        predicted_image_point, is_valid = self.camera.project_point(world_point)
        
        if (is_valid == False):
            #print("[Error_Jacobian]No valid projection")
            return None, None, False 
        
        # (2 x 1)
        error = np.array([
            [predicted_image_point[0] - reference_image_point[0]],
            [predicted_image_point[1] - reference_image_point[1]]
        ])

        #print(f"[Error Jacobian]Error:\n {error}")

        # (x_cam, y_cam, z_cam)
        camera_point = u.w2C(world_point, self.camera.absolutePose())

        # J_icp 
        Jr = np.zeros((3,6))
        Jr[:3, :3] = np.eye(3)
        Jr[:3, 3:6] = u.skew(-camera_point)

        phom = K @ camera_point
        iz = 1./phom[2]
        iz2 = iz ** 2 

        # J_proj
        Jp = np.zeros((2,3))
        Jp[0, :] = [iz, 0, -phom[0] * iz2]
        Jp[1, :] = [0, iz, -phom[1] * iz2]

        # (2 x 6) = (2 x 3) * (3 x 3) * (3 x 6)
        J = Jp @ K @ Jr
      
        return error, J, True
    
    def linearize(self, assoc):
        'Linearize the system and return H and b'
        H = np.zeros((6,6))
        b = np.zeros((6,1))

        for i, (id, best_id) in enumerate(assoc):
            #print(f"[Linearize]id={id},best={best_id}")
            world_point = u.getPoint3D(self.world_points, best_id)
            image_point = self.image_points[i]

            if world_point is None or image_point is None:
                # print("[Linearize]World Point OR Image Point are NONE")
                # print("-------------")
                continue

            error, J, status = self.error_and_jacobian(world_point, image_point)

            if status == False:
                # print("[Linearize][Status=False][Error_Jacobian]Projection of point not valid")
                # print("-------------")
                continue

            # (6 x 2) * (2 x 6) = (6 x 6)
            H += J.T @ J 
            # (6 x 2) * (2 x 1) = (6 x 1)
            b += J.T @ error

        return H, b

    def solve(self, H, b): 
        'Solve a LS problem H*delta_x = -b'
        try:
            return np.linalg.solve(H, -b).reshape(-1, 1)
        except:
            print("[Solve]Singular Matrix")
            return np.zeros((1,6))

    def one_round(self, assoc):
        'Compute dx'
        H, b = self.linearize(assoc)
        #Never singular matrix
        H += np.eye(6)*self.damping
        
        # (1 x 6)
        self.dx = self.solve(H, b).T

        print(f"[Solver]dx: {np.linalg.norm(self.dx)}")
    
    def getMap(self):
        return self.world_points
    
    def points_2d(self):
        return self.image_points
    
    def setMap(self, points3d):
        self.world_points = points3d
    
    def set_image_points(self, image_points):
        self.image_points = image_points