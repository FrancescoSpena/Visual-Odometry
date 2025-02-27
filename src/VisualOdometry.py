import utils as u
import numpy as np
import PICP_solver as solver
import Camera as camera
import matplotlib.pyplot as plt
from collections import Counter

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat'):
        self.camera_info = u.extract_camera_data(camera_path)
        K = self.camera_info['camera_matrix']
        self.cam = camera.Camera(K)
        self.solver = solver.PICP(self.cam)
        self.status = True

    def init(self):
        path0 = u.generate_path(0)
        path1 = u.generate_path(1)

        data_frame_0 = u.extract_measurements(path0)
        data_frame_1 = u.extract_measurements(path1)

        p_0, p_1, _, _, assoc = u.data_association(data_frame_0, data_frame_1)

        #Pose from 0 to 1
        R, t = u.compute_pose(p_0,
                              p_1,
                              self.cam.K)
        
        #3D points
        map = u.triangulate(R,
                            t,
                            p_0,
                            p_1,
                            self.cam.K,
                            assoc)
        
        self.solver.setMap(map)
        T = u.m2T(R,t)
        self.cam.setCameraPose(T)

        #Check
        if(not np.isclose(np.linalg.det(R), 1, atol=1e-6) or np.linalg.norm(t) == 0):
            print(f"det(R): {np.linalg.det(R)}")
            print(f"norm(t): {np.linalg.norm(t)}")
            self.status = False
        
        return self.status
              
    def run(self, idx):
        'Update Relative and Absolute Pose'
        path_prev = u.generate_path(idx-1)
        path_curr = u.generate_path(idx)
        
        prev_frame = u.extract_measurements(path_prev)
        curr_frame = u.extract_measurements(path_curr)

        p_prev, p_curr, _, _, assoc = u.data_association(prev_frame, curr_frame)

        self.cam.updatePrev()
        map = self.solver.getMap()

        #--------------------Debug------------------------

        # print(f"[VisualOdometry]Len of the map: {len(map)}")
        # print(f"[VisualOdometry]Number of point curr: {points_curr.shape[0]}")
        # print(f"[VisualOdometry]Number of associations: {len(assoc)}")

        #---------------------PICP---------------------------

        # At the end of this cicle T_abs is the world express in the
        # reference frame i+1
        for _ in range(3):
            self.solver.initial_guess(self.cam, map, p_curr)
            self.solver.one_round(assoc)
            self.cam.updatePoseICP(self.solver.dx)
        
        self.cam.updateRelative()

        #----------------Update the map------------------------

        #Relative transformation: the image i+1 express in the reference frame of the image i 
        T = self.cam.relativePose()
        R, t = u.T2m(T)

        #Obtain a 3D points of the missing points
        missing_map = u.triangulate(R, t, p_prev, p_curr,
                                    self.cam.cameraMatrix(), assoc)
        
        
        #ID of the points that its already on the map
        id_map = [elem[0] for elem in map]
        #ID of the map between the prev and curr frame
        id_missing_map = [elem[0] for elem in missing_map]

        #ID of the 3D points that are not in the map
        missing = [item for item in id_missing_map if item not in set(id_map)]

        #Take the 3D points that are not in the map and extend the map
        missing_points = []
        for id in missing: 
            point = u.getPoint3D(missing_map, id)
            if(point is not None):
                missing_points.append((id, point))

        #Extend the map with the new points
        map.extend(missing_points)

        #Set the new map
        self.solver.setMap(map)
