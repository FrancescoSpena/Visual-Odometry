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

def T2m(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3,1)
    return R, t

def v2T(v):
    def Rx(roll):
        return np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

    def Ry(pitch):
        return np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    def Rz(yaw):
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

    R = Rx(v[3]) @ Ry(v[4]) @ Rz(v[5])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = v[:3]

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
    
    # for i, j in associations:
    #     print(f"Point {i} in first_data -> Point {j} in second_data")
    #     print(f"  Appearance Features 1:\n {first_data['Appearance_Features'][i]}")
    #     print(f"  Appearance Features 2:\n {second_data['Appearance_Features'][j]}")
    #     print("==================")
    
    points_first = np.array([
        [first_data['Image_X'][i], first_data['Image_Y'][i]]
        for i, _ in associations
    ])
    
    points_second = np.array([
        [second_data['Image_X'][j], second_data['Image_Y'][j]]
        for _, j in associations
    ])
    
    return points_first, points_second, associations

def triangulate(R, t, points1, points2, K):
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points1_hom = points1.T  
    points2_hom = points2.T  

    points4D = cv2.triangulatePoints(P1, P2, points1_hom, points2_hom)
    points3D_hom = points4D / points4D[3]  
    points3D = points3D_hom[:3].T  

    return points3D

def compute_pose(points1, points2, K, z_near=0.0, z_far=5.0):
    E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    
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

def project_point(world_point, camera_matrix, width=640, height=480, z_near=0, z_far=5):
    status = True 
    image_point = np.zeros((2,))
    
    if world_point[2] <= z_near or world_point[2] >= z_far:
        #print("Point out of camera view")
        status = False
    
    projected_point = camera_matrix @ world_point
    
    if np.isclose(projected_point[2], 0):
        #print("Points close to zero")
        status = False
        return image_point, status
    
    image_point[:] = projected_point[:2] / projected_point[2]

    if image_point[0] < 0 or image_point[0] > width-1: 
        #print("Point out of image")
        status = False 
    if image_point[1] < 0 or image_point[1] > height-1:
        #print("Point out of image")
        status = False 

    return image_point, status 

def skew(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def error_and_jacobian(world_point, reference_image_point, K):
    status = True
    predicted_image_point, is_true = project_point(world_point,
                                                    K)
    
    if(is_true == False):
       #print("No good proj")
       status = False
    
    error = predicted_image_point - reference_image_point

    J_r = np.zeros((3,6))
    J_r[:3, :3] = np.eye(3)
    J_r[:3, 3:] = skew(world_point)

    phom = K @ world_point
    
    if np.isclose(phom[2], 0):
        #print("phom close to zero")
        status = False 
    
    iz = 1.0 / phom[2]
    iz2 = iz * iz 

    J_p = np.array([
        [iz, 0, -phom[0]*iz2],
        [0, iz, -phom[1]*iz2]
    ])

    J = J_p @ K @ J_r

    return error, J, status

def linearize(assoc, world_points, reference_image_points, K, kernel_threshold=100):
    status_lin = True
    H = np.zeros((6,6))
    b = np.zeros(6)

    for idx_frame1, idx_frame2 in assoc: 
        if idx_frame1 >= len(reference_image_points) or idx_frame2 >= len(world_points):
            continue  
        
        error, J, status = error_and_jacobian(world_points[idx_frame1], 
                                              reference_image_points[idx_frame2], 
                                              K)

        if status == False:
            #print("Status jacobian false")
            status_lin = False
            continue

        chi = np.dot(error,error)
        lambda_factor = 1.0 

        if chi > kernel_threshold: 
            lambda_factor = np.sqrt(kernel_threshold / chi)

        H += J.T @ J * lambda_factor
        b += J.T @ error * lambda_factor

    return H, b, status_lin

def solve(H, b): 
    try: 
        dx = np.linalg.solve(H, -b)
    except:
        dx = np.zeros_like(b)
    return dx
