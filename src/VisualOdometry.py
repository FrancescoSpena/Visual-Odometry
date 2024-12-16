import utils as u

class VisualOdometry():
    def __init__(self, max_iter=1000, dataset_path='path', camera_path='../data/camera.dat'):
        self.max_iter = max_iter
        self.data = dataset_path
        self.camera_data = u.extract_camera_data(camera_path)

    def run(self):
        #Inizialization (has features of the first and secondo image)

        #Features first and second image 
        first_features = u.extract_measurements(u.generate_path(0))
        second_features = u.extract_measurements(u.generate_path(1))

        #Data Association
        assoc = u.data_association(first_features['Appearance_Features'],
                                   second_features['Appearance_Features'])

        #Point image 1 and 2 
        points1 = [(first_features['Image_X'][i], first_features['Image_Y'][i]) for i, _, _ in assoc]
        points2 = [(second_features['Image_X'][j], second_features['Image_Y'][j]) for _, j, _ in assoc]

        info = u.extract_other_info(u.generate_path(0))
        print(info)
        print("Camera:", self.camera_data)
        
        #New Frame i+1 
        #Feature Extraction (points)
        #Feature Matching (data association)
        #Essential Matrix between image i and i+1
        #Decompose into R_i, t_i and construct T_i (transformation)
        #Relative scale --> rescale translation vector 
        #Compute pose C_i = C_i-1 * T_i
        #Every N iterations apply P-ICP
        pass