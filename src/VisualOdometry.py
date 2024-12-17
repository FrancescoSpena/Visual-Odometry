import utils as u
import numpy as np

class VisualOdometry():
    def __init__(self, max_iter=2, camera_path='../data/camera.dat', traj_path='../data/trajectoy.dat'):
        if max_iter > 120:
            self.max_iter = 120
        else:
            self.max_iter = max_iter
        
        self.camera_info = u.extract_camera_data(camera_path)
        self.gt = u.read_traj(traj_path)
        self.poses_camera = []
        self.R = None
        self.t = None
        self.idx = 0

    def run(self, idx):
        self.idx = idx
        if(idx == 0): 
            #Init
            #Features first and second image 
            first_features = u.extract_measurements(u.generate_path(0))
            second_features = u.extract_measurements(u.generate_path(1))

            #Data Association
            assoc = u.data_association(first_features['Appearance_Features'],
                                    second_features['Appearance_Features'])

            #Point image 1 and 2 
            points1 = np.array([(first_features['Image_X'][i], first_features['Image_Y'][i]) for i, _, _ in assoc])
            points2 = np.array([(second_features['Image_X'][j], second_features['Image_Y'][j]) for _, j, _ in assoc])

            #Compute the essential matrix and decompose in rotation matrix and translation vector
            self.K = self.camera_info['camera_matrix']
            self.R, self.t = u.compute_pose(self.K,points1,points2)
            self.poses_camera.append((self.R,self.t))

            self.all_2d_points = [points1, points2]
            self.all_3d_points = u.triangulate_points(self.K, np.eye(3), np.zeros(3), self.R, self.t, points1, points2)
            self.prev_features = second_features
            return u.m2T(self.R, self.t)

        else:
            curr_features = u.extract_measurements(u.generate_path(self.idx+1))
            curr_assoc = u.data_association(self.prev_features['Appearance_Features'],
                                            curr_features['Appearance_Features'])
        
            prev_points = np.array([(self.prev_features['Image_X'][i], self.prev_features['Image_Y'][i]) for i, _, _ in curr_assoc])
            curr_points = np.array([(curr_features['Image_X'][j], curr_features['Image_Y'][j]) for _, j, _ in curr_assoc])

            R, t = u.compute_pose(self.K,prev_points,curr_points)
            
            #compute scale
            scale = u.getAbsoluteScale(self.gt,self.idx+1)

            #rescale of R and t 
            self.t = self.t + scale * (self.R @ t)
            self.R = R @ self.R
            self.poses_camera.append((self.R, self.t))

            min_points = min(len(self.all_2d_points[-1]), len(self.all_3d_points))
            self.all_2d_points = [points[:min_points] for points in self.all_2d_points]
            self.all_3d_points = self.all_3d_points[:min_points]

            if self.idx % 10 == 0:
                print("Appling BA...")
                self.poses_camera, _ = u.bundle_adjustment(self.K, self.all_2d_points, self.all_3d_points, self.poses_camera)


            self.prev_features = curr_features

            return u.m2T(self.R, self.t)

