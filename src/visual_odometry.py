import numpy as np 
import utils as u
import data_manipulation as data
import testing as test
from scipy.spatial.distance import cosine

camera_info = u.extract_camera_data()
K = camera_info['camera_matrix']

def retriangulation_n_views(map, est_pose, track, measurements_curr):
    """
    Args:
        map (list): {id, (x, y, z)}: 3D points 
        est_pose (list): {pose0, pose1, ...} a list of absolute pose
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        measurements_curr (list): (id, (x,y)) list of measurements
        id_frame: id of the frame
    Returns:
        map: the updated map with new triangulated points
    """

    #Reconstruct the projection matrices from the pose and build the dict
    projection_matrices = []
    for pose in est_pose:
        R, t = u.T2m(pose)
        R = np.round(R, 2)
        t = np.round(t, 2)
        P_curr = K @ np.hstack((R,t))
        projection_matrices.append(P_curr)
    
    dict_projection_matrices = {}
    for i, proj in enumerate(projection_matrices):
        dict_projection_matrices[i] = proj


    #Recover the id of the points to update
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]
    already_in_map = [item for item in id_curr if item in set(id_map)]

    if(len(already_in_map) != 0):
        #Retriangulation of the point with multi view and update the map with the new points
        for id in already_in_map:
            res = extract_measurements_by_id(track, str(id))
            if res:
                new_point = triangulate_n_views(res, dict_projection_matrices) 
                if(new_point is not None):
                    map = u.subPoint(map, id, new_point) 
            else:
                print("no point found")
    
    print("[Retriangulation]All done!")
    return map

def triangulate_n_views(point_tracks, projection_matrices):
    """
    Triangulate a 3D point from multi view using DLT method

    Args:
        point_tracks (dict): { frame_id: (x, y) } 2D points observed in every frame
        projection_matrices (dict): { frame_id: P } projection matrix 3x4 for every frame 

    Returns:
        np.array: 3D coordinates of the reconstructed point [X, Y, Z]
    """
    A = []

    for frame_id, (x, y) in point_tracks.items():
        x, y = float(x), float(y)
        
        if frame_id not in projection_matrices:
            raise ValueError(f"Frame ID {frame_id} not found in projection_matrices.")
        
        P = projection_matrices[frame_id]  # Projection matrix 3x4

        A.append(x * P[2] - P[0]) 
        A.append(y * P[2] - P[1])  

    A = np.array(A) 
    
    _, _, V = np.linalg.svd(A)
    X = V[-1]  
    
    if X[3] == 0:
        raise ValueError("Homogeneous coordinate X[3] is zero. Cannot convert to Cartesian coordinates.")
    
    return X[:3] / X[3]

def extract_measurements_by_id(points_track, id_point):
    """
    Extract the dict that represent the track of the point with id=id_point
    Args: 
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        id_point (str): id of the point
    Return:
        result (dict): that represent the track of the point with id=id_point

    """
    result = {}
    for frame_id, points in points_track.items():
        for point_id, coordinates in points:
            if point_id == id_point:
                result[frame_id] = coordinates
    return result
    
def add_point_to_frame(points_track, frame_id, point_id, point):
    """
    Update the dictionary points_track with the new measure

    Args:
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        frame_id (int): id of the frame
        point_id (str): id of the point
        point (np.array): measure of the point with id=point_id
  
    """
    if frame_id not in points_track:
        points_track[frame_id] = []
        
    point = np.array(point, dtype=np.float32)
    points_track[frame_id].append((point_id, point))

def updateMap(map, measurements_prev, measurements_curr, R, t, T_i):
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]

    #ID of the new no mapped points
    missing = [item for item in id_curr if item not in set(id_map)]

    if(len(missing) != 0):
        #Recover the measure of the prev and curr frame
        prev_points = []
        curr_points = []
        assoc = []
        for elem in missing:
            prev = u.getPoint(measurements_prev, str(elem))
            curr = u.getPoint(measurements_curr, str(elem))

            if(prev is not None and curr is not None):
                prev_points.append(prev)
                curr_points.append(curr)
                assoc.append((elem, elem))
        
        #Triangulation of the missing points w.r.t. the prev frame
        missing_map = data.triangulate(R, t, prev_points, curr_points, K, assoc)

        #Report the points in the global frame and extend the map
        if(len(missing_map) != 0):
            transformed_map = []
            T_i_inv = np.linalg.inv(T_i)

            for id, point in missing_map:
                point_global = test.transform_point(point, T_i_inv)
                transformed_map.append((id, point_global))
    
            map.extend(transformed_map)

            print("[updateMap]All done!")
    
    return map


#------Data association with appearance------


def updateMapApp(map, measurements_prev, measurements_curr, R, t, T_i, assoc):
    """
    Update map with appearance

    Args:
        map (list): (id, point, app) map with 3D points
        measurements_prev (list): (id, point, app) 
        measurements_curr (list): (id, point, app)
        R (3x3): rotatiom matrix from prev to curr frame 
        t (3x1): translation vector from prev to curr frame
        T_i (4x4): transformation 0_T_prev
        assoc (list): (id_prev, best_curr)
    Return: 
        map (list): the updated map

    """

    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]

    #ID of the new no-mapped points
    missing = [item for item in id_curr if item not in set(id_map)]

    if(len(missing) != 0):
        prev_points = []
        curr_points = []
        assoc_missing = []
        app_missing = []

        for id_miss in missing:
            prev = u.getPointApp(measurements_prev, str(id_miss))
            curr = u.getPointApp(measurements_curr, str(id_miss))
            app = u.getApp(measurements_curr, str(id_miss))
            id = u.getId(assoc, str(id_miss))

            
            if(prev is not None and curr is not None and app is not None and id is not None):
                prev_points.append(prev)
                curr_points.append(curr)
                app_missing.append(app)
                assoc_missing.append((id, id_miss))

            
        missing_map = data.triangulateWithApp(R=R, t=t, points1=prev_points, 
                                              points2=curr_points, K=K, assoc=assoc_missing,
                                              app_curr_frame=app_missing)
        
        if(len(missing_map) != 0):
            transformed_map = []
            T_i_inv = np.linalg.inv(T_i)

            for id, point, app in missing_map:
                point_global = test.transform_point(point, T_i_inv)
                transformed_map.append((id, point_global, app))
            
            map.extend(transformed_map)
            print("[UpdateMap]All done!")
    
    return map

def retriangulation_n_views_app(map, est_pose, track, measurements_curr):
    """
    Args:
        map (list): (id, (x, y, z), app): 3D points 
        est_pose (list): {pose0, pose1, ...} a list of absolute pose
        points_tracks (dict): {frame_id: list(id_point,(x,y)} 2D points observed in every frame
        measurements_curr (list): (id, (x,y), app) list of measurements
        id_frame: id of the frame
    Returns:
        map: the updated map with new triangulated points
    """

    #Reconstruct the projection matrices from the pose and build the dict
    projection_matrices = []
    for pose in est_pose:
        R, t = u.T2m(pose)
        R = np.round(R, 2)
        t = np.round(t, 2)
        P_curr = K @ np.hstack((R,t))
        projection_matrices.append(P_curr)
    
    dict_projection_matrices = {}
    for i, proj in enumerate(projection_matrices):
        dict_projection_matrices[i] = proj


    #Recover the id of the points to update
    id_map = [item[0] for item in map]
    id_curr = [item[0] for item in measurements_curr]
    already_in_map = [item for item in id_curr if item in set(id_map)]

    if(len(already_in_map) != 0):
        #Retriangulation of the point with multi view and update the map with the new points
        for id in already_in_map:
            res = extract_measurements_by_id(track, str(id))
            if res:
                new_point = triangulate_n_views(res, dict_projection_matrices) 
                if(new_point is not None):
                    map = u.subPointApp(map, id, new_point) 
            else:
                print("no point found")
    
    print("[Retriangulation]All done!")
    return map


def data_association_frame(points_prev, points_curr):
    """
    Args:
        points_prev (list): (id, point, app)
        points_curr (list): (id_curr, point_curr, app_curr)
    Return:
        assoc (list): (id, best)
    """

    assoc = []

    def compute_similarity(appearance_first, appearance_second):
        appearance_first = list(map(float, appearance_first))
        appearance_second = list(map(float, appearance_second))
        return 1 - cosine(appearance_first, appearance_second)
    
    for id, _, app in points_prev:
        best_sim = -1
        best_id = None

        for id_curr, _, app_curr in points_curr:  
            sim = compute_similarity(app, app_curr)

            if sim > best_sim:
                best_sim = sim 
                best_id = id_curr
            
        if best_id is not None: 
            assoc.append((id, best_id))

    return assoc