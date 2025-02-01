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
    started = False

    with open(file_path, 'r') as f: 
        for line in f: 
            line = line.strip()
            if not line: 
                continue
            
            if line.startswith("point"):
                tokens = line.split()
                # Expected tokens:
                # tokens[0]: "point"
                # tokens[1]: POINT_ID_CURRENT_MEASUREMENT
                # tokens[2]: ACTUAL_POINT_ID
                # tokens[3]: IMAGE_X coordinate
                # tokens[4]: IMAGE_Y coordinate
                # tokens[5:]: APPEARANCE features
                if(len(tokens) < 5):
                    continue

                point_id = tokens[1]
                actual_id = tokens[2]
                image_x = tokens[3]
                image_y = tokens[4]
                appearance = tokens[5:]

                point_data.append({
                    'Point_ID': point_id,
                    'Actual_ID': actual_id,
                    'Image_X': image_x,
                    'Image_Y': image_y,
                    'Appearance_Features': appearance
                })
            

    df_points = pd.DataFrame(point_data)

    result = {
        'Point_IDs': df_points['Point_ID'].tolist(),
        'Actual_IDs': df_points['Actual_ID'].tolist(),
        'Image_X': df_points['Image_X'].tolist(),
        'Image_Y': df_points['Image_Y'].tolist(),
        'Appearance_Features': df_points['Appearance_Features'].tolist()
    }
    
    return result


def old_extract_measurements(file_path):
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

def data_association(first_data, second_data):
    '''
    return {
        'Point_IDs': df_points['Point_ID'].tolist(),
        'Actual_IDs': df_points['Actual_ID'].tolist(),
        'Image_X': df_points['Image_X'].tolist(),
        'Image_Y': df_points['Image_Y'].tolist(),
        'Appearance_Features': df_points['Appearance_Features'].tolist()
    }

    Point_IDs è un indice progressivo che indica quante misure hai in una singola immagine
    
    Actual_IDs è un indice UNIVOCO considerando tutti i punti nel mondo
    
    Appearance descrive il punto nell'immagine. Idealmente hai la stessa appearance ogni volta che misuri lo STESSO punto.
    In pratica non succede e bisogna prendere quello a minimum distance (cosine similarity)
    '''

    actual_id_first_frame = first_data['Actual_IDs']
    point_id = first_data['Point_IDs']
    num_points_first_data = len(point_id)
    appearance_first_data = first_data['Appearance_Features']

    actual_id_second_frame = second_data['Actual_IDs']
    point_id_second_frame = second_data['Point_IDs']
    num_points_second_data = len(point_id_second_frame)
    appearance_second_data = first_data['Appearance_Features']
    assoc = []

    min_distance = float('inf')

    id_temp_first_frame = -1 
    id_temp_second_frame = -1

    print(f"num points first data: {num_points_first_data}")
    print(f"num points second data: {num_points_second_data}")

    print(f"len list app first data: {len(appearance_first_data)}")
    print(f"len list app second data: {len(appearance_second_data)}")

    for i in range(0, num_points_first_data):
        a = np.array(list(map(float, appearance_first_data[i])))
        for j in range(0, num_points_second_data): 
            b = np.array(list(map(float, appearance_second_data[j])))

            cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
            distance = 1 - cosine_similarity

            if distance < min_distance:
                min_distance = distance
                id_temp_first_frame = actual_id_first_frame[i]
                id_temp_second_frame = actual_id_second_frame[j]
        
        assoc.append((id_temp_first_frame, id_temp_second_frame))
        min_distance = float('inf')
        
    return assoc 

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
        if count_in_front >= max_points_in_front:
            max_points_in_front = count_in_front
            best_R, best_t = R_cand, t_cand

    return best_R, best_t


def makePoints(data_frame0, data_frame1, assoc):
    '''
    Return two numpy array p0 = (x0, y0), p1 = (x1, y1) with the i-esim element of p0 is the coord. of
    the point with app_id = id and i-esim element of p1 is the coord. of the point with app_id = best_id
    '''
    p0 = []
    p1 = []

    for elem in assoc:
            id, best = elem
            point0 = get_point(data_frame0, id)
            point1 = get_point(data_frame1, best)

            if(point0 is not None and point1 is not None):
                p0.append(point0)
                p1.append(point1)

    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)

    return p0, p1

def get_point(data, id):
    actual_id = data['Actual_IDs']
    point_id = data['Point_IDs']
    num_points = len(point_id)
    x = data['Image_X']
    y = data['Image_Y']

    for i in range(0, num_points):
        actual = actual_id[i]
        if(actual == id):
            return x[i], y[i]
    
    return None

def getPoint3D(points, id):

    for id_point, point in points:
        if(id_point == id):
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

