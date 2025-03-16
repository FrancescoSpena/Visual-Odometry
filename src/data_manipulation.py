import numpy as np
import cv2
from scipy.spatial.distance import cosine


def getMeasurementsFromDataFrame(first_data, second_data):
    'Return p0, p1, points_first=(ID, (x, y)), points_second=(ID, (x,y))'
    point_id_first_data = first_data['Point_IDs']
    actual_id_first_data = first_data['Actual_IDs']
    coord_x_first = first_data['Image_X']
    coord_y_first = first_data['Image_Y']

    point_id_second_data = second_data['Point_IDs']
    actual_id_second_data = second_data['Actual_IDs']
    coord_x_second = second_data['Image_X']
    coord_y_second = second_data['Image_Y']

    points_first = []
    points_second = []

    for i in range(len(point_id_first_data)):
        act_first = actual_id_first_data[i]
        for j in range(len(point_id_second_data)):
            act_second = actual_id_second_data[j]

            if(act_first == act_second):
                xfirst = coord_x_first[i]
                yfirst = coord_y_first[i]
                xsecond = coord_x_second[j]
                ysecond = coord_y_second[j]
                points_first.append((act_first, (xfirst, yfirst)))
                points_second.append((act_second, (xsecond, ysecond)))
                

    return points_first, points_second

def association3d(map, points_frame_curr, camera):
    points_proj = []
    assoc_3d = []

    for elem in map:
        id, point = elem 
        point_proj, isvalid = camera.project_point(point)
        if(isvalid):
            points_proj.append((id, point_proj))
        
    for elem_proj in points_proj:
        id_proj, point_proj = elem_proj
        for elem_curr in points_frame_curr:
            id_curr, _ = elem_curr
            if id_proj == id_curr:
                assoc_3d.append((id_proj, id_curr))

    return assoc_3d

def data_association(first_data, second_data):
    'Return p0, p1, points_first=(ID, (x, y)), points_second=(ID, (x,y)), assoc=(ID, best)'
    point_id_first_data = first_data['Point_IDs']
    actual_id_first_data = first_data['Actual_IDs']
    coord_x_first = first_data['Image_X']
    coord_y_first = first_data['Image_Y']

    point_id_second_data = second_data['Point_IDs']
    actual_id_second_data = second_data['Actual_IDs']
    coord_x_second = second_data['Image_X']
    coord_y_second = second_data['Image_Y']

    points_first = []
    points_second = []
    assoc = []

    for i in range(len(point_id_first_data)):
        act_first = actual_id_first_data[i]
        for j in range(len(point_id_second_data)):
            act_second = actual_id_second_data[j]

            if(act_first == act_second):
                xfirst = coord_x_first[i]
                yfirst = coord_y_first[i]
                xsecond = coord_x_second[j]
                ysecond = coord_y_second[j]
                points_first.append((act_first, (xfirst, yfirst)))
                points_second.append((act_second, (xsecond, ysecond)))
                assoc.append((act_first, act_second))
                
    p0 = [item[1] for item in points_first]
    p1 = [item[1] for item in points_second]

    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)

    return p0, p1, points_first, points_second, assoc

def triangulate(R, t, points1, points2, K, assoc):
    'Return a list of 3d points (ID, (X, Y, Z))'
    'assoc = (ID, best_ID)'

    assert len(points1) == len(points2) == len(assoc)

    # print(f"R:\n {R}, \nt:\n {t}")
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # print(f"P1:\n {P1}")
    # print(f"P2:\n {P2}")

    points1 = np.array(points1).reshape(-1, 2).T  # (2, N)
    points2 = np.array(points2).reshape(-1, 2).T  # (2, N)

    # print(f"points1: {points1.shape}")
    # print(f"points2: {points2.shape}")

    points4D = cv2.triangulatePoints(P1, P2, points1, points2)

    points4D /= points4D[3]    # x /= w
    points3D = points4D[:3].T  # (N x 3)

    id_points3D = []
    ids = [pair[0] for pair in assoc]
    for i, point in enumerate(points3D):
        id_points3D.append((ids[i], point))

    return id_points3D

def check_essential(E):
    U, S, Vt = np.linalg.svd(E)

    rank_valid = np.isclose(S[2], 0, atol=1e-6) and np.isclose(S[0], S[1], atol=1e-3)
    det_valid = np.isclose(np.linalg.det(E), 0, atol=1e-6)

    return rank_valid and det_valid, {"singular_values": S, "determinant": np.linalg.det(E)}

def compute_pose(points1_frame, points2_frame, K):
    'Compute E -> Pose'

    if not points1_frame or not points2_frame:
        raise ValueError("Input points cannot be empty")

    points1 = [item[1] for item in points1_frame]
    points2 = [item[1] for item in points2_frame]

    points1 = np.array(points1, dtype=np.float32)
    points2 = np.array(points2, dtype=np.float32)
    
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    
    valid, info = check_essential(E)
    if not valid:
        raise ValueError(f"Invalid Essential Matrix: {info}")

    
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
        points4D /= points4D[3]  # Normalize homogeneous coordinates
        
        return countPointsInFront(points4D, P2)

    best_R, best_t = np.eye(3), np.zeros((3,1))
    max_points_in_front = -1
    
    for R_cand, t_cand in poss_sol:
        count_in_front = bestSolution(R_cand, t_cand, K, points1, points2)
        #print(f"[Estimate pose]Count in front: {count_in_front}")
        if count_in_front >= max_points_in_front:
            #print("[Estimate pose]Find new")
            max_points_in_front = count_in_front
            best_R, best_t = R_cand, t_cand

    return best_R, best_t

#------Function with appearance------
def data_association_with_similarity(first_data, second_data):
    'Return p0, p1, points_first=(ID, (x, y), app), points_second=(ID, (x,y), app), assoc=(ID, best)'
    
    def compute_similarity(appearance_first, appearance_second):
        appearance_first = list(map(float, appearance_first))
        appearance_second = list(map(float, appearance_second))
        return (1 - cosine(appearance_first, appearance_second))
    
    point_id_first_data = first_data['Point_IDs']
    actual_id_first_data = first_data['Actual_IDs']
    coord_x_first = first_data['Image_X']
    coord_y_first = first_data['Image_Y']
    appearence_first_data = first_data['Appearance_Features']

    point_id_second_data = second_data['Point_IDs']
    actual_id_second_data = second_data['Actual_IDs']
    coord_x_second = second_data['Image_X']
    coord_y_second = second_data['Image_Y']
    appearence_second_data = second_data['Appearance_Features']

    points_first = []
    points_second = []
    assoc = []

    for i in range(len(point_id_first_data)):
        act_first = actual_id_first_data[i]
        app_first = appearence_first_data[i]
        
        best_match = None
        best_match_act = None
        best_sim = -1 
        best_x_second = None
        best_y_second = None
        best_app = None
        
        for j in range(len(point_id_second_data)):
            act_second = actual_id_second_data[j]
            app_second = appearence_second_data[j]
            sim = compute_similarity(app_first, app_second)

            if(sim > best_sim):
                best_sim = sim 
                best_match = j
                best_match_act = act_second
                best_x_second = coord_x_second[j]
                best_y_second = coord_y_second[j]
                best_app = app_second
            
        if best_match is not None:
            xfirst, yfirst = coord_x_first[i], coord_y_first[i]
            points_first.append((act_first, (xfirst, yfirst), app_first))
            points_second.append((best_match_act, (best_x_second, best_y_second), best_app))
            assoc.append((act_first, best_match))

                
    p0 = [item[1] for item in points_first]
    p1 = [item[1] for item in points_second]

    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)

    return p0, p1, points_first, points_second, assoc

def getMeasurementsFromDataFrameApp(first_data, second_data):
    'Return p0, p1, points_first=(ID, (x, y), app), points_second=(ID, (x,y), app)'
    point_id_first_data = first_data['Point_IDs']
    actual_id_first_data = first_data['Actual_IDs']
    coord_x_first = first_data['Image_X']
    coord_y_first = first_data['Image_Y']
    appearence_first_data = first_data['Appearance_Features']


    point_id_second_data = second_data['Point_IDs']
    actual_id_second_data = second_data['Actual_IDs']
    coord_x_second = second_data['Image_X']
    coord_y_second = second_data['Image_Y']
    appearence_second_data = second_data['Appearance_Features']


    points_first = []
    points_second = []

    for i in range(len(point_id_first_data)):
        act_first = actual_id_first_data[i]
        for j in range(len(point_id_second_data)):
            act_second = actual_id_second_data[j]

            if(act_first == act_second):
                xfirst = coord_x_first[i]
                yfirst = coord_y_first[i]
                xsecond = coord_x_second[j]
                ysecond = coord_y_second[j]
                app_first = appearence_first_data[i]
                app_second = appearence_second_data[j]
                points_first.append((act_first, (xfirst, yfirst), app_first))
                points_second.append((act_second, (xsecond, ysecond), app_second))
                

    return points_first, points_second

def triangulateWithApp(R, t, points1, points2, K, assoc, app):
    'Return a list of 3d points (ID, (X, Y, Z))'
    'assoc = (ID, best_ID)'

    assert len(points1) == len(points2) == len(assoc) == len(app)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    points1 = np.array(points1).reshape(-1, 2).T  # (2, N)
    points2 = np.array(points2).reshape(-1, 2).T  # (2, N)

    points4D = cv2.triangulatePoints(P1, P2, points1, points2)

    points4D /= points4D[3]    # x /= w
    points3D = points4D[:3].T  # (N x 3)

    id_points3D = []
    ids = [pair[0] for pair in assoc]
    for i, point in enumerate(points3D):
        id_points3D.append((ids[i], point, app[i]))

    return id_points3D

def association3d_with_similarity(map_points, points_frame_curr, camera):
    """
    Return the association vector between the map and the measure of the curr frame

    Args:
        map_points (list): (id, point, app) the map that contains the 3D points 
        points_frame_curr (list): (id, point, app) the measurements of the curr frame
        camera (obj.Camera): the camera object

    Return: 
        association vector
    """
    def compute_similarity(appearance_first, appearance_second):
        appearance_first = list(map(float, appearance_first))
        appearance_second = list(map(float, appearance_second))
        return (1 - cosine(appearance_first, appearance_second))
    
    points_proj = []
    assoc3d_app = []
    
    #Project the map into the image plane
    for elem in map_points: 
        id, point, app = elem 
        proj, isvalid = camera.project_point(point)
        if(isvalid):
            points_proj.append((id, proj, app))
    
    for elem in points_proj:
        id, point, app = elem 

        best_match = None 
        best_sim = -1

        for id_curr, point_curr, app_curr in points_frame_curr:
            sim = compute_similarity(app, app_curr)

            if(sim > best_sim):
                best_sim = sim 
                best_match = id_curr
        
        if best_match is not None:
            assoc3d_app.append((id, best_match))

    return assoc3d_app



#------Function with appearance------