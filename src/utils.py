import pandas as pd 
import re
from scipy.spatial.distance import euclidean
import numpy as np
import cv2
from scipy.optimize import least_squares

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

def data_association(first_data, second_data):
    id_map = dict(zip(first_data['Point_IDs'], first_data['Actual_IDs']))
    
    associations = []
    seen = set()  
    
    for i, point_id in enumerate(second_data['Point_IDs']):
        actual_id_second = second_data['Actual_IDs'][i]
        
        if point_id in id_map and id_map[point_id] == actual_id_second:
            key = (point_id, actual_id_second)  
            
            if key not in seen:
                associations.append({
                    'Point_ID_First': point_id,
                    'Point_ID_Second': point_id,
                    'Actual_ID': actual_id_second,
                    'Image_X_First': first_data['Image_X'][first_data['Point_IDs'].index(point_id)],
                    'Image_Y_First': first_data['Image_Y'][first_data['Point_IDs'].index(point_id)],
                    'Image_X_Second': second_data['Image_X'][i],
                    'Image_Y_Second': second_data['Image_Y'][i],
                    'Appearance_First': first_data['Appearance_Features'][first_data['Point_IDs'].index(point_id)],
                    'Appearance_Second': second_data['Appearance_Features'][i],
                })
                seen.add(key)  
    
    return associations

def filter_3d_points(points, z_near, z_far):
    return np.array([
        point for point in points
        if z_near < point[2] < z_far and point[2] > 0
    ])

def compute_pose(points1, points2, K, z_near=0.1, z_far=100.0):
    points1 = np.array(points1)
    points2 = np.array(points2)

    if len(points1) < 5 or len(points2) < 5:
        raise ValueError("No 5 points!")

    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)

    if E is None:
        raise ValueError("No E found.")

    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    R1, R2, t = cv2.decomposeEssentialMat(E)

    possible_poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    points1_norm = cv2.undistortPoints(np.expand_dims(points1, axis=1), K, None)
    points2_norm = cv2.undistortPoints(np.expand_dims(points2, axis=1), K, None)

    def is_valid_pose(R, t):
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        points4D = cv2.triangulatePoints(P1, P2, points1_norm, points2_norm)
        points4D[3, points4D[3] == 0] = 1e-10  # Evita divisioni per zero

        points3D = points4D[:3] / points4D[3]

        valid_points = 0
        for point in points3D.T:
            if z_near < point[2] < z_far:
                valid_points += 1

        return valid_points / len(points3D.T) > 0.75

    for R, t in possible_poses:
        if is_valid_pose(R, t):
            return R, t

    raise ValueError("Nessuna soluzione valida trovata")

    

def bundle_adjustment(camera_matrix, points_2d_list, points_3d_list, poses):
    pass



