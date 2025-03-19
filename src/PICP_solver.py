import numpy as np 
import utils as u


class PICP():
    def __init__(self, camera):
        self.camera = camera 
        self.world_points = None 
        self.image_points = None
        self.damping = 0.0001
    
    def initial_guess(self, camera, world_points, point_curr_frame):
        """
        Set the map and image points in a ref values
        Args:
            camera (obj.Camera): camera object
            world_points (list): (id, (x,y,z)) 
            point_curr_frame (list): (id, (x,y))
        Return:
            None
        """

        self.world_points = world_points
        self.image_points = point_curr_frame
        self.camera = camera
    
    def error_and_jacobian(self, world_point, reference_image_point):
        """
        Compute error and jacobian given the 3D point and 2D point

        Args:
            world_point (list): (id, (x,y,z))
            reference_image_point (list): (id, (x,y))
        Return:
            error, J, flag (2x1) (2x6) (bool): error, jacobian and flag=True if all good
        """

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
        """
        Linearize the system applied GS alghorithm

        Args:
            assoc (list): (id, best_id)
        Return:
            H, b (6x6) (6x1): matrix to make Ax = b
        """
        H = np.zeros((6,6))
        b = np.zeros((6,1))

        no_valid_world = 0
        no_valid_image = 0

        for i, (id, best_id) in enumerate(assoc):
            world_point = u.getPoint3D(self.world_points, id)
            image_point = u.getPoint(self.image_points, best_id)

            if world_point is None:
                no_valid_world +=1
                continue
            if image_point is None: 
                no_valid_image += 1
                continue

            error, J, status = self.error_and_jacobian(world_point, image_point)

            if status == False:
                continue

            # (6 x 2) * (2 x 6) = (6 x 6)
            H += J.T @ J 
            # (6 x 2) * (2 x 1) = (6 x 1)
            b += J.T @ error

        return H, b

    def solve(self, H, b): 
        """
        Solve a LS problem H*dx = -b

        Args:
            H (6x6): matrix H
            b (6x1): matrix B
        Return: 
            dx (1x6): euclidean perturbation
        """
        try:
            return np.linalg.solve(H, -b).reshape(-1, 1)
        except:
            print("[Solve]Singular Matrix")
            return np.zeros((1,6))

    def one_round(self, assoc):
        """
        Compute the perturbation given the measurements and the associations

        Args:
            assoc (list): (id, best)
        Return: 
            None --> Update the data structure
        """

        H, b = self.linearize(assoc)
        #Never singular matrix
        H += np.eye(6)*self.damping
        
        # (1 x 6)
        self.dx = self.solve(H, b).T

        #Print the norm of the perturbation
        print(f"[Solver]dx: {np.linalg.norm(self.dx)}")
    
    def getMap(self):
        """
        Return the map
        Args:
            None
        Return:
            map (list): (id, (x,y,z))
        """
        return self.world_points
    
    def points_2d(self):
        """
        Return the 2D points

        Args:
            None 
        Return:
            image_points (list): (id, (x,y))
        """
        return self.image_points
    
    def setMap(self, points3d):
        """
        Set the map in the data structure
        Args:
            points3d (list): (id, (x,y,z))
        Return:
            None
        """
        self.world_points = points3d
    
    def set_image_points(self, image_points):
        """
        Set the image points in the data structure 
        Args:
            image_points (list): (id, (x,y))
        Return: 
            None
        """
        self.image_points = image_points