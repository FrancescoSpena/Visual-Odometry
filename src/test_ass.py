import utils as u
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt

import Camera as cam
import VisualOdometry as vo 
import random
import pandas as pd 

#Generate 3D points with structure (ID, (x, y, z))
def generate_3d_points(num_points, x_range=640, y_range=480, z_range=3):
    points = []
    for i in range(1, num_points + 1):
        x = random.uniform(0, x_range)
        y = random.uniform(0, y_range)
        z = random.uniform(1, z_range)
        points.append((i, (x, y, z)))
    return points


#Generate data frame:
def generate_dataframe(num_points):
    data = {
        'Point_ID': [i + 1 for i in range(num_points)],
        'Actual_ID': [random.randint(1000, 9999) for _ in range(num_points)],
        'Image_X': [random.uniform(0.0, 640.0) for _ in range(num_points)],  # Assuming a 640x480 image resolution
        'Image_Y': [random.uniform(0.0, 480.0) for _ in range(num_points)],
        'Appearance_Features': [list(np.random.rand(5)) for _ in range(num_points)],  # 5 random features per point
    }
    
    df_points = pd.DataFrame(data)
    
    result = {
        'Point_IDs': df_points['Point_ID'].tolist(),
        'Actual_IDs': df_points['Actual_ID'].tolist(),
        'Image_X': df_points['Image_X'].tolist(),
        'Image_Y': df_points['Image_Y'].tolist(),
        'Appearance_Features': df_points['Appearance_Features'].tolist(),
    }
    
    return result

#Function to apply a transformation (R, t) on the 3D points and return an np.array with
#structure (ID, (x, y, z))
def transform_3d_points(points, R, t):
    transformed_points = []

    for point in points:
        point_id, (x, y, z) = point
        original_point = np.array([x, y, z]).reshape(3, 1)
        transformed_point = (R @ original_point + t).flatten()
        transformed_points.append((point_id, (transformed_point[0], transformed_point[1], transformed_point[2])))

    return transformed_points

#Function to apply a transformation (R, t) on the 2D points with structure (ID, (x, y))
def transform_2d_points(points, R, t):
    transformed_points = []

    for point in points:
        point_id, (x, y) = point
        original_point = np.array([x, y]).reshape(2, 1)
        transformed_point = (R @ original_point + t).flatten()
        transformed_points.append((point_id, (transformed_point[0], transformed_point[1])))

    return transformed_points

def create_2d_points_from_frame(frame):
    point_ids = frame['Point_IDs']
    image_x = frame['Image_X']
    image_y = frame['Image_Y']

    structured_points = np.array(
        [(pid, (x, y)) for pid, x, y in zip(point_ids, image_x, image_y)],
        dtype=[('ID', int), ('Coordinates', float, (2,))]
    )
    
    return structured_points

def create_new_frame(frame, points): 
    new_ids = [point[0] for point in points]
    new_x = [point[1][0] for point in points]
    new_y = [point[1][1] for point in points]
    
    # Create a new dictionary by updating Point_IDs, Image_X, and Image_Y
    updated_result = frame.copy()
    updated_result['Point_IDs'] = new_ids
    updated_result['Image_X'] = new_x
    updated_result['Image_Y'] = new_y

    return updated_result

def plot_associations(frame1, frame2, points1, points2):
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



#Generate data frame 
frame1 = generate_dataframe(50)

#Extract 2D points from frame 
points_2d_frame1 = create_2d_points_from_frame(frame1)

#Apply transformation 
angle_radians = 30
R = np.array([
    [np.cos(angle_radians), -np.sin(angle_radians)],
    [np.sin(angle_radians),  np.cos(angle_radians)]
])
t = np.zeros((2,1))

points_2d_frame2 = transform_2d_points(points_2d_frame1, R, t)

#Create new frame
frame2 = create_new_frame(frame1, points_2d_frame1)

#Data association
points1, points2, assoc = u.data_association(frame1, frame2)


