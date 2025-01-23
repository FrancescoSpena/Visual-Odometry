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

def gt2T(gt):
    'gt to homogeneous transformation'
    T = np.eye(4)
    T[:3, 3] = gt 
    return T

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

    T = np.eye(4, dtype=np.float32)  
    T[:3, :3] = Rx(v[3]) @ Ry(v[4]) @ Rz(v[5])
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
    
    # for i, j in associations:
    #     print(f"Point {i} in first_data -> Point {j} in second_data")
    #     print(f"  Appearance Features 1:\n {first_data['Appearance_Features'][i]}")
    #     print(f"  Appearance Features 2:\n {second_data['Appearance_Features'][j]}")
    #     print("==================")
    
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
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points1_hom = points1.T  
    points2_hom = points2.T  

    points4D = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)
    points3D_hom = points4D / points4D[3]  
    points3D = points3D_hom[:3].T  

    points3D_with_indices = [(assoc[idx][0], points3D[idx]) for idx in range(len(points3D))]

    return points3D_with_indices

def compute_pose(points1, points2, K, diff_gt, z_near=0.0, z_far=5.0):
    'Compute E -> Pose'
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)

    def triangulate_and_check(R, t, points1, points2, K, z_near=0.0, z_far=5.2):
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        
        points1_hom = points1.T  # Shape: (2, N)
        points2_hom = points2.T  # Shape: (2, N)
        points4D = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)
        points3D = points4D / points4D[3]  # (x, y, z)
        
        z_values = points3D[2]

        cheirality_check = np.all(z_values > 0)
        tolerance = 1e-2
        depth_check = np.all((z_values >= z_near) & (z_values <= z_far + tolerance))

        return cheirality_check and depth_check

    poss_sol = [(R, t), (R, -t), (R.T, t), (R.T, -t)]
    
    for i, (R_candidate, t_candidate) in enumerate(poss_sol):
        vo_dist = np.linalg.norm(t_candidate)
        scale = diff_gt / vo_dist
        t_candidate *= scale

        if triangulate_and_check(R_candidate, t_candidate, points1, points2, K, z_near, z_far):
            return R_candidate, t_candidate

    print("No sol.")
    return np.eye(3), np.zeros((3, 1))

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

def skew(vector):
    'Convert a vector in a skew-symmetric matrix'
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def w2C(world_point, camera_pose):
    'From world point to camera point'

    world_point_h = np.append(world_point, 1)
    camera_point = camera_pose @ world_point_h
    camera_point = camera_point[:3] / camera_point[3]

    return camera_point