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

def data_association(points1, points2):
    associations = []
    
    for i, p1 in enumerate(points1):
        best_match = None
        min_distance = float('inf')
        
        for j, p2 in enumerate(points2):
            distance = euclidean(p1, p2)
            if distance < min_distance:
                min_distance = distance
                best_match = j
        
        associations.append((i, best_match, min_distance))
    
    return associations

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

def compute_pose(K, points1, points2):
    E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    return R, t

def read_traj(trajectory_path):
    ground_truths = []

    with open(trajectory_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            if len(values) >= 7:
                x, y, z = map(float, values[4:7])
                ground_truths.append((x, y, z))
    
    return ground_truths


def getAbsoluteScale(gt, frame_id):
    if frame_id >= len(gt):
        return 1.0  #Default 

    x_prev, y_prev, z_prev = gt[frame_id - 1]
    x_curr, y_curr, z_curr = gt[frame_id]

    scale = np.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2 + (z_curr - z_prev) ** 2)
    return scale

def bundle_adjustment(camera_matrix, width, height, points_2d_list, points_3d_list, poses):
    pass


def triangulate_points(K, R1, t1, R2, t2, points1, points2, z_near=None, z_far=None):
    P1 = K @ np.hstack((R1, t1.reshape(-1, 1)))
    P2 = K @ np.hstack((R2, t2.reshape(-1, 1)))
    
    points1_h = np.array(points1).T
    points2_h = np.array(points2).T
    points_4d = cv2.triangulatePoints(P1, P2, points1_h, points2_h)
    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d


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

def normalize_3d_points(points_3d_list):
    normalized_points = []
    for points in points_3d_list:
        points_array = np.array(points)
        if points_array.ndim == 2 and points_array.shape[1] == 3:
            normalized_points.append(points_array)
        else:
            print(f"Warning: Skipping invalid 3D points with shape {points_array.shape}")
    return normalized_points
