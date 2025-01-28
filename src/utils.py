import pandas as pd 
import re
from scipy.spatial.distance import euclidean
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.distance import cosine
from scipy.spatial import KDTree


def extract_measurements(file_path):
    'Read measurements file'
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
    'Read other file'
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
    'Generate path_measurements from ID'
    return f"../data/meas-{id:05d}.dat"

def extract_camera_data(file_path):
    'Read camera file'
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

def m2T(R, t):
    'From R,t to homogeneous transformation'
    t = t.reshape(3,1)
    T = np.eye(4)
    T[:3, :3] = R 
    T[:3, 3] = t.ravel()
    return T

def g2T(v):
    'From (x, y, theta) to homogeneous transformation'
    x = v[0]
    y = v[1]
    theta = v[2]
    T = np.eye(4)

    c = np.cos(theta)
    s = np.sin(theta)
    
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

    t = np.array([
        [x],
        [y],
        [0]
    ])

    T = np.block([
        [R, t],
        [np.array([0, 0, 0, 1])]
    ])

    return T

def relativeMotion(T0, T1):
    'relative motion between two homogenous transformation'
    'e.g. T1 = 0_T_1, T2 = 0_T_2 --> 1_T_2 = inv(0_T_1) @ 0_T_2'
    return np.linalg.inv(T0) @ T1

def errorEstimatedToGt(T_gt_rel, T_est_rel):
    return np.linalg.inv(T_est_rel) @ T_gt_rel

def T2m(T):
    'From homogeneous transformation to R, t'
    R = T[:3, :3]
    t = T[:3, 3].reshape(3,1)
    return R, t

def v2T(v):
    'From vector to homogeneous transformation'
    def Rx(roll):
        c = np.cos(roll)
        s = np.sin(roll)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    def Ry(pitch):
        c = np.cos(pitch)
        s = np.sin(pitch)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def Rz(yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def angles2R(angles):
        roll, pitch, yaw = angles
        R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
        return R

    v = np.array(v).flatten()
    T = np.eye(4)  
    T[:3, :3] = angles2R(v[3:])  
    T[:3, 3] = v[:3] 
    return T

def plot_match(points1, points2):
    'Plot points'
    import matplotlib.pyplot as plt
    
    # Extract 2D points and IDs from (id, point_2d)
    points1_2d = np.array([p[1] for p in points1])
    points2_2d = np.array([p[1] for p in points2])
    ids1 = [p[0] for p in points1]
    ids2 = [p[0] for p in points2]

    plt.figure()
    plt.scatter(points1_2d[:, 0], points1_2d[:, 1], label="Frame 1")
    plt.scatter(points2_2d[:, 0], points2_2d[:, 1], label="Frame 2")

    # Annotate IDs for Frame 1
    for i, txt in enumerate(ids1):
        plt.annotate(txt, (points1_2d[i, 0], points1_2d[i, 1]), fontsize=8, color='blue')

    # Annotate IDs for Frame 2
    for i, txt in enumerate(ids2):
        plt.annotate(txt, (points2_2d[i, 0], points2_2d[i, 1]), fontsize=8, color='orange')

    plt.legend()
    plt.title("Matched Points")
    plt.show()

def data_association(first_data, second_data, threshold=0.2):
    'Perform data association, return point1 = (ID, (X,Y)), point2 = (ID, (X,Y)), assoc = (ID, best_ID)'
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

    points_first = [(i, np.array([first_data['Image_X'][i], first_data['Image_Y'][i]])) for i, _ in associations]
    points_second = [(j, np.array([second_data['Image_X'][j], second_data['Image_Y'][j]])) for _, j in associations]

    return points_first, points_second, associations

def read_traj(path='../data/trajectory.dat'):
    'Read the trajectory file'
    gt = []
    with open(path, 'r') as file: 
        for line in file: 
            if line.strip() and not line.startswith('#'):
                elements = line.split()
                gt_pose = list(map(float, elements[4:7]))  # [x, y, z]
                gt.append(gt_pose)
    return gt

def triangulate(R, t, points1, points2, K, assoc):
    'Return a list of 3d points (ID, (X, Y, Z))'
    'assoc = (ID, best_ID)'
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points1 = points1.T  
    points2 = points2.T  

    points4D = cv2.triangulatePoints(P1, P2, points1, points2)
    
    points4D /= points4D[3]    # x /= w
    points3D = points4D[:3].T  # (N x 3)


    id_points3D = []
    ids = [pair[0] for pair in assoc]
    for i, point in enumerate(points3D):
        id_points3D.append((ids[i], point))

    return id_points3D

def compute_pose(points1, points2, K):
    'Compute E -> Pose'
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    
    points1 = points1[mask.ravel() == 1]    
    points2 = points2[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

    poss_sol = [(R, t), (R, -t), (R.T, t), (R.T, -t)]

    def bestSolution(R, t, K, points1, points2):
        def countPointsInFront(points4D, P):
            points3D_cam = P @ points4D #(x_cam, y_cam, z_cam)
            depths = points3D_cam[2]
            count = 0
            for d in depths: 
                if (d > 0):
                    count += 1
            return count

        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))

        points1 = points1.T
        points2 = points2.T

        points4D = cv2.triangulatePoints(P1, P2, points1, points2)
        points4D /= points4D[3]

        return countPointsInFront(points4D, P2)

        
    best_R, best_t = np.eye(3), np.zeros((3,1))
    max_points_in_front = -1
    
    for R_cand, t_cand in poss_sol:
        count_in_front = bestSolution(R_cand, t_cand, K, points1, points2)
        if count_in_front > max_points_in_front:
            max_points_in_front = count_in_front
            best_R, best_t = R_cand, t_cand

    return best_R, best_t


def update_point(vector, target_idx, new_point):
    'Update point with ID=target_idx'
    for i in range(len(vector)):
        if vector[i][0] == target_idx:
            vector[i] = (vector[i][0], new_point)
            return True 
    return False

def get_point(vector, target_idx):
    'Return the point with the ID=target_idx'
    for idx, point in vector: 
        if idx == target_idx:
            return point
    return None

def skew(v):
    'Convert a vector in a skew-symmetric matrix'
    return np.array([
        [  0,   -v[2],  v[1]],
        [ v[2],    0,  -v[0]],
        [-v[1],  v[0],    0]
    ])

def w2C(world_point, camera_pose):
    'Return p_cam'

    world_point = np.append(world_point, 1)

    p_cam = camera_pose @ world_point

    return p_cam[:3]

