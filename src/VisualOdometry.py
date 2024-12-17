import utils as u
import numpy as np

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat', traj_path='../data/trajectoy.dat'):
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

    #Return a transformation between frame i and i+1
    def run(self, idx):
        self.idx = idx

        if idx == 0: 
            # Init
            first_features = u.extract_measurements(u.generate_path(0))
            second_features = u.extract_measurements(u.generate_path(1))

            # Data Association
            assoc = u.data_association(first_features['Appearance_Features'],
                                    second_features['Appearance_Features'])

            # Punti immagine 1 e 2
            points1 = np.array([(first_features['Image_X'][i], first_features['Image_Y'][i]) for i, _, _ in assoc])
            points2 = np.array([(second_features['Image_X'][j], second_features['Image_Y'][j]) for _, j, _ in assoc])

            # Calcolo della posa (R e t)
            self.R, self.t = u.compute_pose(self.K, points1, points2)
            self.poses_camera.append((self.R, self.t))

            # Triangolazione dei punti 3D
            R1, t1 = np.eye(3), np.zeros((3, 1))
            R2, t2 = self.R, self.t
            points_3d = u.triangulate_points(self.K, R1, t1, R2, t2, points1, points2, self.z_near, self.z_far)

            self.all_2d_points.append(points2)
            self.all_3d_points.append(points_3d)
            
            self.prev_features = second_features
            return u.m2T(self.R, self.t)

        else:
            curr_features = u.extract_measurements(u.generate_path(self.idx + 1))
            curr_assoc = u.data_association(self.prev_features['Appearance_Features'],
                                            curr_features['Appearance_Features'])

            # Punti immagine precedente e corrente
            prev_points = np.array([(self.prev_features['Image_X'][i], self.prev_features['Image_Y'][i]) for i, _, _ in curr_assoc])
            curr_points = np.array([(curr_features['Image_X'][j], curr_features['Image_Y'][j]) for _, j, _ in curr_assoc])

            # Calcolo della posa (R e t)
            R, t = u.compute_pose(self.K, prev_points, curr_points)
            
            # Calcolo della scala
            scale = u.getAbsoluteScale(self.gt, self.idx + 1)

            # Aggiornamento di R e t
            self.t = self.t + scale * (self.R @ t)
            self.R = R @ self.R
            self.poses_camera.append((self.R, self.t))

            #Update 2d/3d points
            R1, t1 = np.eye(3), np.zeros((3, 1))
            R2, t2 = self.R, self.t
            points_3d = u.triangulate_points(self.K, R1, t1, R2, t2, points1, points2, self.z_near, self.z_far)

            self.all_2d_points.append(points2)
            self.all_3d_points.append(points_3d)

            if self.idx % 30 == 0:
                print(f"idx = {self.idx} - Applying BA...")
                self.poses_camera, self.all_3d_points = u.bundle_adjustment(self.K, 
                                                                            self.width, 
                                                                            self.height, 
                                                                            self.all_2d_points, 
                                                                            self.all_3d_points, 
                                                                            self.poses_camera)


            self.prev_features = curr_features
            return u.m2T(self.R, self.t)


