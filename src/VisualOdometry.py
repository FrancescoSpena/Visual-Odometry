import utils as u
import numpy as np

class VisualOdometry():
    def __init__(self, max_iter=10, camera_path='../data/camera.dat', traj_path='../data/trajectoy.dat'):
        self.max_iter = max_iter
        self.camera_info = u.extract_camera_data(camera_path)
        self.gt = u.read_traj(traj_path)
        self.poses_camera = []
        self.R = None
        self.t = None

    def run(self):
        #Inizialization (has features of the first and second image)

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
        K = self.camera_info['camera_matrix']
        self.R, self.t = u.compute_pose(K,points1,points2)
        self.poses_camera.append((self.R,self.t))

        print(f"Rotation:\n {self.R}")
        print(f"translation:\n {self.t}")
        print("=============1=================")
        prev_features = second_features

        #Repeat the previous steps for [2,max_iter]
        for i in range(2,self.max_iter):
            curr_features = u.extract_measurements(u.generate_path(i))
            curr_assoc = u.data_association(prev_features['Appearance_Features'],
                                            curr_features['Appearance_Features'])

            prev_points = np.array([(prev_features['Image_X'][i], prev_features['Image_Y'][i]) for i, _, _ in curr_assoc])
            curr_points = np.array([(curr_features['Image_X'][j], curr_features['Image_Y'][j]) for _, j, _ in curr_assoc])

            R, t = u.compute_pose(K,prev_points,curr_points)
            
            #compute scale
            scale = u.getAbsoluteScale(self.gt,i)

            #rescale of R and t 
            self.t = self.t + scale * (self.R * t)
            self.R = R * self.R
            self.poses_camera.append((self.R, self.t))
            
            print(f"Rotation: \n {self.R}")
            print(f"translation: \n {self.t}")
            print(f"============{i}==================")

            #Every 10 iterations apply P-ICP
            if(i % 10 == 0):
                pass

            prev_features = curr_features