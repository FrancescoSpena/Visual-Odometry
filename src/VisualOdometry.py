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
        self.prev_frame = None

    def init(self):
        path0 = u.generate_path(0)
        path1 = u.generate_path(1)

        data_frame_0 = u.extract_measurements(path0)
        data_frame_1 = u.extract_measurements(path1)
        assoc = u.data_association(data_frame_0, data_frame_1)

        p_0, p_1 = u.makePoints(data_frame_0, data_frame_1, assoc)
        
         
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
        
        T_init = u.m2T(R,t)
        self.cam.setCameraPose(T_init)
        self.solver.set_map(map)
        self.prev_frame = data_frame_1

        #Check
        if(not np.isclose(np.linalg.det(R), 1, atol=1e-6) or np.linalg.norm(t) == 0):
            print(f"det(R): {np.linalg.det(R)}")
            print(f"norm(t): {np.linalg.norm(t)}")
            self.status = False
        
        return self.status
              
    def run(self, idx):
        'Update relative and absolute pose'
        path_curr = u.generate_path(idx)
        
        prev_frame = self.prev_frame
        curr_frame = u.extract_measurements(path_curr)

        assoc = u.data_association(prev_frame,curr_frame)

        points_prev, points_curr = u.makePoints(prev_frame, curr_frame, assoc)


        for _ in range(10):
            self.solver.initial_guess(self.cam, self.solver.getMap(), points_prev)
            self.solver.one_round(assoc)
            self.cam.updatePose(self.solver.dx)

        # T = self.cam.absolutePose()
        # R, t = u.T2m(T)
        # map = u.triangulate(R, t, points_prev, points_curr, self.cam.cameraMatrix(), assoc)
        
        # self.solver.set_map(map)
        # self.prev_frame = curr_frame



def test_assoc(points1, points2, assoc):
    if points1.shape != points2.shape or points1.shape[0] != len(assoc):
        raise ValueError("points1, points2, and assoc must have the same length")
    
    plt.figure(figsize=(12, 8))

    for i, (p1, p2) in enumerate(zip(points1, points2)):
        id1, id2 = assoc[i]  # Extract IDs from assoc
        
        # Draw dashed line
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='dashed', color='black', alpha=0.2)
        
        # Display IDs near the points
        plt.text(p1[0], p1[1], f"{id1}", fontsize=8, color='red', verticalalignment='bottom', horizontalalignment='right')
        plt.text(p2[0], p2[1], f"{id2}", fontsize=8, color='blue', verticalalignment='bottom', horizontalalignment='left')

    # Plot points
    plt.scatter(points1[:, 0], points1[:, 1], color='red', label='points1')
    plt.scatter(points2[:, 0], points2[:, 1], color='blue', label='points2')

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Dashed Line Associations with IDs")
    plt.grid(True)
    plt.show()


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

    for (id, _), (point) in zip(assoc, points):
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