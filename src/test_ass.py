import utils as u
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt


def generate_frame(num_points=50, noise_std=0.01):
    points = np.random.rand(num_points, 2)
    point_ids = list(range(num_points))
    actual_ids = point_ids.copy()  
    appearance_features = np.random.rand(num_points, 128)

    return {
        'Point_IDs': point_ids,
        'Actual_IDs': actual_ids,
        'Image_X': points[:, 0].tolist(),
        'Image_Y': points[:, 1].tolist(),
        'Appearance_Features': appearance_features
    }

def transform_frame(frame, translation, rotation_angle, noise_std=0.01):
    points = np.array([frame['Image_X'], frame['Image_Y']]).T
    
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    
    rotated_points = points @ rotation_matrix.T
    transformed_points = rotated_points + np.array(translation)
    transformed_points += np.random.normal(0, noise_std, transformed_points.shape)
    
    return {
        'Point_IDs': frame['Point_IDs'],
        'Actual_IDs': frame['Actual_IDs'],
        'Image_X': transformed_points[:, 0].tolist(),
        'Image_Y': transformed_points[:, 1].tolist(),
        'Appearance_Features': frame['Appearance_Features']
    }

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


frame1 = generate_frame()

translation = [0.5, -0.1]
rotation_angle = np.pi / 6 #30°

frame2 = transform_frame(frame1, translation, rotation_angle)

points1, points2, associations = u.data_association(frame1, frame2)

plot_associations(frame1, frame2, points1, points2)
