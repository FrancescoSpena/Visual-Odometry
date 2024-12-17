import utils as u
import numpy as np

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat', traj_path='../data/trajectoy.dat', optim=50):
        self.camera_info = u.extract_camera_data(camera_path)
        self.K = self.camera_info['camera_matrix']
        self.width = self.camera_info['width']
        self.height = self.camera_info['height']
        self.z_near = self.camera_info['z_near']
        self.z_far = self.camera_info['z_far']
        
        self.gt = u.read_traj(traj_path)
        self.poses_camera = []
        self.R = None
        self.t = None
        self.idx = 0

        self.all_2d_points = []
        self.all_3d_points = []
        self.optim = optim

    # Return a transformation between frame i and i+1
    def run(self, idx):
        self.idx = idx

        if idx == 0: 
            # Init
            first_features = u.extract_measurements(u.generate_path(0))
            second_features = u.extract_measurements(u.generate_path(1))

            # Data Association
            assoc = u.data_association(first_features['Appearance_Features'],
                                       second_features['Appearance_Features'])
            
            # Point image 1 and 2 
            points1 = np.array([(first_features['Image_X'][i], first_features['Image_Y'][i]) for i, _, _ in assoc])
            points2 = np.array([(second_features['Image_X'][j], second_features['Image_Y'][j]) for _, j, _ in assoc])

            # Compute pose
            self.R, self.t = u.compute_pose(self.K, points1, points2)
            self.poses_camera.append((self.R, self.t))

            self.prev_features = second_features
            return u.m2T(self.R, self.t)

        else:
            curr_features = u.extract_measurements(u.generate_path(self.idx + 1))
            curr_assoc = u.data_association(self.prev_features['Appearance_Features'],
                                            curr_features['Appearance_Features'])
            
            # Points image previous and current 
            prev_points = np.array([(self.prev_features['Image_X'][i], self.prev_features['Image_Y'][i]) for i, _, _ in curr_assoc])
            curr_points = np.array([(curr_features['Image_X'][j], curr_features['Image_Y'][j]) for _, j, _ in curr_assoc])

            # Compute pose
            R, t = u.compute_pose(self.K, prev_points, curr_points)
            
            # Compute scale
            scale = u.getAbsoluteScale(self.gt, self.idx + 1)

            # Update R and t
            self.t = self.t + scale * (self.R @ t)
            self.R = R @ self.R
            self.poses_camera.append((self.R, self.t))

            # Append points for bundle adjustment if the dimensions match
            triangulated_points = u.triangulate_points(self.K, 
                                                       self.R, 
                                                       self.t, 
                                                       R, 
                                                       t, 
                                                       prev_points, 
                                                       curr_points)
            
            self.all_2d_points.append(curr_points)
            self.all_3d_points.append(triangulated_points)

            max_points = 1000  
            if len(self.all_3d_points) > max_points:
                self.all_3d_points = self.all_3d_points[-max_points:]
                self.all_2d_points = self.all_2d_points[-max_points:]


            # Call bundle adjustment every 15 iterations
            if (self.idx + 1) % self.optim == 0:
                print("Calling BA...")
                self.poses_camera, optimized_points_3d = u.bundle_adjustment(self.K, 
                                                                             self.all_2d_points, 
                                                                             self.all_3d_points, 
                                                                             self.poses_camera)
                self.all_3d_points = [optimized_points_3d]

            self.prev_features = curr_features
            return u.m2T(self.R, self.t)
