import utils as u
import numpy as np
import PICP_solver as solver
import Camera as camera
import matplotlib.pyplot as plt

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

        #points0: frame 0, points1: frame1 --> (id, (x,y))
        points0, points1, assoc = u.data_association(data_frame_0, 
                                                     data_frame_1)
        
        #Extract only (x, y)
        p_0 = np.array([item[1] for item in points0])
        p_1 = np.array([item[1] for item in points1])

        #Pose from 0 to 1
        R, t = u.compute_pose(p_0,
                              p_1,
                              self.cam.K)
        
        
        #3D points of the frame 0
        map = u.triangulate(R,
                            t,
                            p_0,
                            p_1,
                            self.cam.K,
                            assoc)
        
        T_init = u.m2T(R,t)
        self.cam.setCameraPose(T_init)
        self.solver.set_map(map)
        self.prev_frame = data_frame_1

        #Check
        if(np.linalg.det(R) != 1 or np.linalg.norm(t) == 0):
            self.status = False
        
        return self.status
    
    def picp(self, assoc):
        pass

            
    def run(self, idx):
        'Update relative and absolute pose'
        path_curr = u.generate_path(idx)
        
        prev_frame = self.prev_frame
        curr_frame = u.extract_measurements(path_curr)

        points_prev, points_curr, assoc = u.data_association(prev_frame,curr_frame)

        #test(self.cam, world_points=self.solver.map(), points=points_curr, assoc=assoc)
        
        for _ in range(1, 300):
            self.solver.initial_guess(self.cam, self.solver.getMap(), points_prev)
            self.solver.one_round(assoc)
            self.cam.updatePose(self.solver.dx)


        #test(self.cam, world_points=self.solver.map(), points=points_curr, assoc=assoc)

        
        # I want to transformation from frame i and frame i+1
        
        # T = self.cam.relativePose()
        # R, t = u.T2m(T)
        
        # p_0 = np.array([item[1] for item in points_prev])
        # p_1 = np.array([item[1] for item in points_curr])
        
        # update_map = u.triangulate(R, t, p_0, p_1, self.cam.cameraMatrix(), assoc)
        # self.solver.set_map(update_map)
        
        
        self.prev_frame = curr_frame



def test(cam, world_points, points, assoc):
    image_points = []
    ids = []

    curr = []
    ids_curr = []

    # world_points = (id, point) but the id is refered to the frame i
    # take the 3D point and project it in the frame i+1 with id = best_id, because in this frame the 2D point
    # associate with the 3D point in the frame i has id in the frame i+1 equal to best_id
    for (_, point), (_, best_id) in zip(world_points, assoc):
        point_in_the_image, is_valid = cam.project_point(point)
        if(is_valid):
            image_points.append(point_in_the_image)
            ids.append(best_id)

    for id, point in points:
        curr.append(point)
        ids_curr.append(id)

    image_points = np.array(image_points)
    curr = np.array(curr)

    plt.figure(figsize=(12, 8))
    
    plt.scatter(image_points[:, 0], image_points[:, 1], c='red', label='Projected Points')
    for idx, (x, y) in zip(ids, image_points):
        plt.text(x, y, str(idx), color='blue', fontsize=8)
    
    plt.scatter(curr[:, 0], curr[:, 1], c='green', label='Current Points')
    for idx, (x, y) in zip(ids_curr, curr):
        plt.text(x, y, str(idx), color='orange', fontsize=8)
    
    for id_proj, (x_proj, y_proj) in zip(ids, image_points):
        for id_curr, (x_curr, y_curr) in zip(ids_curr, curr):
            if id_proj == id_curr:  # Match points with the same ID
                plt.plot([x_proj, x_curr], [y_proj, y_curr], 'k--', linewidth=0.8)  # Dashed line

    plt.title('Projected and Current Points on Image Plane')
    plt.xlabel('X (Image)')
    plt.ylabel('Y (Image)')
    plt.gca().invert_yaxis()  # Invert Y-axis for image coordinates
    plt.legend()
    plt.grid(True)
    plt.show()