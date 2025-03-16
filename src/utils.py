import pandas as pd 
import numpy as np
import cv2
from scipy.spatial.distance import cosine

def extract_measurements(file_path):
    point_data = []

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

def generate_path(id):
    'Generate path_measurements from ID'
    return f"../data/meas-{id:05d}.dat"

def extract_world_data(file_path='../data/world.dat'):
    world_info = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if(len(parts) < 4):
                continue

            landmark_id = parts[0]
            position = [float(x) for x in parts[1:4]]
            appearance = parts[4:]

            world_info[landmark_id] = {
                "position": position,
                "appearance": appearance
            }
    
    return world_info

def extract_camera_data(file_path='../data/camera.dat'):
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

def Rx(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
    ])

def Ry(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def Rz(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def v2T(v):
    'From vector to homogeneous transformation'
    def angles2R(angles):
        roll, pitch, yaw = angles
        R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
        return R

    v = np.array(v).flatten()
    T = np.eye(4)  
    T[:3, :3] = angles2R(v[3:])  
    T[:3, 3] = v[:3] 
    return T

def homogeneous_rotation(R):
    H = np.eye(4)
    H[:3, :3] = R
    return H

def alignWithWorldFrame(T_cam):
    'From c_T_w to w_T_c'
    T_cam_cp = np.copy(T_cam)
    T_cam_cp = np.linalg.inv(T_cam_cp)

    theta = np.deg2rad(90)

    # Combined rotation (first x, then y)
    R = Ry(theta) @ Rz(-theta)
    H_R = homogeneous_rotation(R)
    T_cam_cp = H_R @ T_cam_cp @ H_R.T

    return T_cam_cp

def alignWithCameraFrame(T_world):
    'From w_T_c to c_T_w'
    T_world_cp = np.copy(T_world)
    theta = np.deg2rad(90)
    R = Rz(theta) @ Ry(-theta)
    H_R = homogeneous_rotation(R)
    
    T_world_cp = H_R @ T_world_cp @ H_R.T 
    T_world_cp = np.linalg.inv(T_world_cp)

    return T_world_cp

def read_traj(path='../data/trajectory.dat'):
    'Read the trajectory file'
    gt = []
    with open(path, 'r') as file: 
        for line in file: 
            parts = line.strip().split()
            if len(parts) >= 7:
                gt_pose = list(map(float, parts[4:7]))
                gt.append(gt_pose)
    return gt

def getPoint3D(point_frame, target_id):
    for id, point in point_frame:
        if(id == target_id):
            return point
    return None

def getPoint(point_frame, target_id):
    'Return 2D given the id'
    'type(target_id) = str'
    for elem in point_frame:
        id, point = elem 

        if(id == target_id):
            point = [float(x) for x in point]
            point = np.array(point)
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

    #p_cam = c_T_w @ w_p
    p_cam = camera_pose @ world_point

    return p_cam[:3]

def subPoint(map, target_id, new_point):
    for i, (id, point) in enumerate(map):
        if id == target_id:
            map[i] = tuple((id, new_point))
            return map
    return map

#------Function with appearance------
def getPoint3DApp(point_frame, target_id):
    'Return 3D point given the id'
    'type(target_id) = str'
    for id, point, _ in point_frame:
        if(id == target_id):
            return point
    return None

def getPointApp(point_frame, target_id):
    'Return 2D point given the id'
    'type(target_id) = str'
    for elem in point_frame:
        id, point, _ = elem 

        if(id == target_id):
            point = [float(x) for x in point]
            point = np.array(point)
            return point
    
    return None

def subPointApp(map, target_id, new_point):
    'Sub the point with id=target_id with new_point'
    for i, (id, point, app) in enumerate(map):
        if id == target_id:
            map[i] = tuple((id, new_point, app))
            return map
    return map

def getBest(assoc, target_id):
    'Return the best_id given the target in assoc vector'
    for id, best in assoc:
        if(id == target_id):
            return best
    return None

def getApp(map, target_id):
    'Return the appearance of the point with id=target_id'
    for id, _, app in map: 
        if(id == target_id):
            return app
    return None
#------Function with appearance------