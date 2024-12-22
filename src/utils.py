import pandas as pd 
import re
from scipy.spatial.distance import euclidean
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.distance import cosine
from scipy.spatial import KDTree


def extract_measurements(file_path):
    point_data = []
    pattern = re.compile(r'point\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(.+)')

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = pattern.match(line)
            if match:
                point_id = int(match.group(1))
                actual_id = int(match.group(2))
                image_x = float(match.group(3))
                image_y = float(match.group(4))
                appearance = [float(val) for val in match.group(5).split()]
                
                point_data.append({
                    'Point_ID': point_id,
                    'Actual_ID': actual_id,
                    'Image_X': image_x,
                    'Image_Y': image_y,
                    'Appearance_Features': appearance
                })

    df_points = pd.DataFrame(point_data)

    return {
        'Point_IDs': df_points['Point_ID'].tolist(),
        'Actual_IDs': df_points['Actual_ID'].tolist(),
        'Image_X': df_points['Image_X'].tolist(),
        'Image_Y': df_points['Image_Y'].tolist(),
        'Appearance_Features': df_points['Appearance_Features'].tolist()
    }

def extract_other_info(file_path):
    sequences = []
    ground_truths = []
    odometry_poses = []

    seq_pattern = re.compile(r'seq:\s*(\d+)')
    gt_pose_pattern = re.compile(r'gt_pose:\s*([\d\-.e]+)\s+([\d\-.e]+)\s+([\d\-.e]+)')
    odom_pose_pattern = re.compile(r'odom_pose:\s*([\d\-.e]+)\s+([\d\-.e]+)\s+([\d\-.e]+)')

    with open(file_path, 'r') as file:
        for line in file:
            seq_match = seq_pattern.search(line)
            if seq_match:
                sequences.append(int(seq_match.group(1)))

            gt_match = gt_pose_pattern.search(line)
            if gt_match:
                ground_truths.append(tuple(map(float, gt_match.groups())))

            odom_match = odom_pose_pattern.search(line)
            if odom_match:
                odometry_poses.append(tuple(map(float, odom_match.groups())))

    return {
        'Sequences': sequences,
        'Ground_Truths': ground_truths,
        'Odometry_Poses': odometry_poses
    }

def generate_path(id):
    return f"../data/meas-{id:05d}.dat"

def extract_camera_data(file_path):
    camera_matrix = []
    cam_transform = []
    z_near, z_far = None, None
    width, height = None, None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        reading_camera_matrix = False
        reading_cam_transform = False
        
        for line in lines:
            line = line.strip()

            if line == "camera matrix:":
                reading_camera_matrix = True
                continue
            if reading_camera_matrix and line != "":
                if len(camera_matrix) < 3:
                    camera_matrix.append([float(val) for val in line.split()])
                if len(camera_matrix) == 3:
                    reading_camera_matrix = False

            if line == "cam_transform:":
                reading_cam_transform = True
                continue
            if reading_cam_transform and line != "":
                if len(cam_transform) < 4:
                    cam_transform.append([float(val) for val in line.split()])
                if len(cam_transform) == 4:
                    reading_cam_transform = False

            if line.startswith("z_near:"):
                z_near = float(line.split(":")[1].strip())
            if line.startswith("z_far:"):
                z_far = float(line.split(":")[1].strip())
            if line.startswith("width:"):
                width = int(line.split(":")[1].strip())
            if line.startswith("height:"):
                height = int(line.split(":")[1].strip())

    camera_matrix = np.array(camera_matrix) if camera_matrix else None
    cam_transform = np.array(cam_transform) if cam_transform else None

    return {
        "camera_matrix": camera_matrix,
        "cam_transform": cam_transform,
        "z_near": z_near,
        "z_far": z_far,
        "width": width,
        "height": height
    }

def read_traj(trajectory_path):
    ground_truths = []

    with open(trajectory_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) >= 7:
                x, y, z = map(float, values[4:7])
                ground_truths.append((x, y, z))
    
    return ground_truths

def m2T(R, t):
    t = t.reshape(3,1)
    T = np.eye(4)
    T[:3, :3] = R 
    T[:3, 3] = t.ravel()
    return T

def gt2T(gt):
    T = np.eye(4)
    T[:3, 3] = gt 
    return T

def plot_match(points1, points2):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(points1[:, 0], points1[:, 1], label="Frame 1")
    plt.scatter(points2[:, 0], points2[:, 1], label="Frame 2")
    plt.legend()
    plt.title("Matched Points")
    plt.show()

def data_association(first_data, second_data, threshold=0.2):
    associations = []
    
    for i, actual_id1 in enumerate(first_data['Actual_IDs']):
        candidate_indices = [j for j, actual_id2 in enumerate(second_data['Actual_IDs']) if actual_id1 == actual_id2]
        
        if len(candidate_indices) == 1:
            associations.append((i, candidate_indices[0]))
        elif len(candidate_indices) > 1:
            min_distance = float('inf')
            best_match_idx = None
            
            for j in candidate_indices:
                distance = cosine(first_data['Appearance_Features'][i], second_data['Appearance_Features'][j])
                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = j
            
            if min_distance < threshold:
                associations.append((i, best_match_idx))
    
    points_first = np.array([
        [first_data['Image_X'][i], first_data['Image_Y'][i]]
        for i, _ in associations
    ])
    
    points_second = np.array([
        [second_data['Image_X'][j], second_data['Image_Y'][j]]
        for _, j in associations
    ])
    
    return points_first, points_second

def triangulate(R, t, points1, points2, K):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points1_hom = points1.T  # Shape: (2, N)
    points2_hom = points2.T  # Shape: (2, N)

    points4D = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)
    points3D_hom = points4D / points4D[3]  # Normalizza per ottenere coordinate omogenee
    points3D = points3D_hom[:3].T  # Converte in forma Nx3

    return points3D


def estimate_normals(target_points, k=10):
    target_points = np.hstack((target_points, np.zeros((target_points.shape[0], 1))))

    kd_tree = KDTree(target_points)
    normals = []

    for p in target_points:
        num_points = len(target_points)
        k_adjusted = min(k + 1, num_points)

        _, indices = kd_tree.query(p, k=k_adjusted)

        neighbors = target_points[indices[1:]]

        mean = np.mean(neighbors, axis=0)
        covariance_matrix = np.zeros((3, 3))
        for n in neighbors:
            diff = n - mean
            covariance_matrix += np.outer(diff, diff)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        normal = eigenvectors[:, np.argmin(eigenvalues)]

        if np.dot(normal, p) > 0:
            normal = -normal

        normals.append(normal)

    return np.array(normals)

def compute_pose(points1, points2, K, z_near=0.0, z_far=5.0):
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

    def triangulate_and_check(R, t, points1, points2, K, z_near=0.0, z_far=5.0):
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        
        points1_hom = points1.T  # Shape: (2, N)
        points2_hom = points2.T  # Shape: (2, N)
        points4D = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)
        points3D = points4D / points4D[3]  # (x, y, z, 1)
        
        z_values = points3D[2]
        
        cheirality_check = np.all(z_values > 0)
        depth_check = np.all((z_values >= z_near) & (z_values <= z_far))
        
        return cheirality_check and depth_check

    poss_sol = [(R, t), (R, -t), (R.T, t), (R.T, -t)]
    
    for i, (R_candidate, t_candidate) in enumerate(poss_sol):
        scale_factor = 0.01  
        t_scaled = t_candidate * scale_factor

        if triangulate_and_check(R_candidate, t_scaled, points1, points2, K, z_near, z_far):
            return R_candidate, t_candidate

    print("No sol.")
    return np.eye(3), np.zeros((3, 1))

def transform_point_in_dict(point_data, R, t):

    image_x_list = point_data['Image_X']
    image_y_list = point_data['Image_Y']

    transformed_x = []
    transformed_y = []

    for x,y in zip(image_x_list,image_y_list):
        point_3d = np.array([x,y,0.0])
        transformed_point = (R @ point_3d + t).flatten()

        transformed_x.append(transformed_point[0])
        transformed_y.append(transformed_point[1])

    point_data['Image_X'] = transformed_x
    point_data['Image_Y'] = transformed_y

    return point_data

def compute_errors(transformed_points, target_points, target_normals):
    transformed_points = np.hstack((transformed_points, np.zeros((transformed_points.shape[0], 1))))
    target_points = np.hstack((target_points, np.zeros((target_points.shape[0], 1))))
    
    errors = []

    for i in range(len(transformed_points)):
        p_transformed = transformed_points[i]
        p_target = target_points[i]
        n_target = target_normals[i]

        error = ((p_transformed - p_target) @ n_target)
        errors.append(error)

    return errors


def compute_jacobian_and_residuals(transformed_points, target_points, target_normals):
    transformed_points = np.hstack((transformed_points, np.zeros((transformed_points.shape[0], 1))))
    target_points = np.hstack((target_points, np.zeros((target_points.shape[0], 1))))
    
    J = []
    residuals = []

    for i in range(len(transformed_points)):
        p_transformed = transformed_points[i]
        n_target = target_normals[i]

        J_rot = np.cross(p_transformed, n_target)  
        J_trans = n_target                         

        J.append(np.hstack((J_rot, J_trans)))

        residual = np.dot((p_transformed - target_points[i]), n_target)
        residuals.append(residual)
    
    J = np.array(J)  
    residuals = np.array(residuals)  

    return J, residuals

import numpy as np

def rotation_from_vector(w):
    theta = np.linalg.norm(w) 
    
    if theta == 0:
        return np.eye(3)
    
    w_hat = w / theta
    
    K = np.array([
        [0, -w_hat[2], w_hat[1]],
        [w_hat[2], 0, -w_hat[0]],
        [-w_hat[1], w_hat[0], 0]
    ])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def update_pose(R, t, delta_pose):
    delta_R = rotation_from_vector(delta_pose[:3])
    delta_t = delta_pose[3:].reshape(3,1)

    R = delta_R @ R 
    t = t + delta_t
    return R, t