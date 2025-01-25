import utils as u
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt

import Camera as cam
import VisualOdometry as vo 

def generate_frame(num_points=50, noise_std=0.01, depth_range=(2.0, 8.0)):
    points_2d = np.random.rand(num_points, 2)

    depths = np.random.uniform(depth_range[0], depth_range[1], size=num_points)
    points_3d = np.hstack((points_2d, depths[:, np.newaxis]))

    points_3d += np.random.normal(0, noise_std, points_3d.shape)

    point_ids = list(range(num_points))
    actual_ids = point_ids.copy()
    appearance_features = np.random.rand(num_points, 128)

    frame_data = {
        'Point_IDs': point_ids,
        'Actual_IDs': actual_ids,
        'Image_X': points_2d[:, 0].tolist(),
        'Image_Y': points_2d[:, 1].tolist(),
        'Appearance_Features': appearance_features
    }

    points_3d_data = {
        'Point_IDs': point_ids,
        'Points_3D': points_3d.tolist()
    }

    return frame_data, points_3d_data

def transform_frame(frame, translation, rotation_angle, noise_std=0.01, points_3d=None):
    points = np.array([frame['Image_X'], frame['Image_Y']]).T

    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])

    rotated_points = points @ rotation_matrix.T
    transformed_points = rotated_points + np.array(translation[:2])
    transformed_points += np.random.normal(0, noise_std, transformed_points.shape)

    transformed_points_3d = None
    if points_3d is not None:
        
        rotation_matrix_3d = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])

        transformed_points_3d = []
        for point in points_3d:
            transformed_coords = (rotation_matrix_3d @ point['Coordinates']).T + np.array(translation)
            transformed_coords += np.random.normal(0, noise_std, transformed_coords.shape)
            transformed_points_3d.append({'ID': point['ID'], 'Coordinates': transformed_coords})

    transformed_frame = {
        'Point_IDs': frame['Point_IDs'],
        'Actual_IDs': frame['Actual_IDs'],
        'Image_X': transformed_points[:, 0].tolist(),
        'Image_Y': transformed_points[:, 1].tolist(),
        'Appearance_Features': frame['Appearance_Features']
    }

    return transformed_frame, transformed_points_3d



def plot_associations(frame1, frame2, points1, points2):
    plt.figure(figsize=(10, 6))
    plt.scatter(frame1['Image_X'], frame1['Image_Y'], c='blue', label='Frame 1')
    plt.scatter(frame2['Image_X'], frame2['Image_Y'], c='red', label='Frame 2')
    for (id1, point1), (id2, point2) in zip(points1, points2):
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--', alpha=0.5)
    plt.legend()
    plt.title("Data Associations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


frame1, points_3d_frame1 = generate_frame()

translation = [0.5, -0.1]
rotation_angle = np.pi / 6 #30°

frame2, points_3d_frame2 = transform_frame(frame1, translation, rotation_angle, points_3d_frame1)

points0, points1, assoc = u.data_association(frame1, frame2)

plot_associations(frame1, frame2, points0, points1)


camera_path='../data/camera.dat'
camera_info = u.extract_camera_data(camera_path)
camera_matrix = camera_info['camera_matrix']
camera = cam.Camera(camera_matrix)



# vo.test(assoc, camera, points_3d_frame2, points1)