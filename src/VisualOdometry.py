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
        tolerance = 1e-2
        while(True):
            self.solver.one_round(assoc)
            T_rel = u.v2T(self.solver.dx)
            self.cam.updatePose(T_rel)

            dx = self.solver.dx
            dx_norm = np.linalg.norm(dx)
            print(f"dx: {dx_norm}")
            if(dx_norm < tolerance):
                break

            
    def run(self, idx):
        'Update relative and absolute pose'
        path_curr = u.generate_path(idx)
        
        prev_frame = self.prev_frame
        curr_frame = u.extract_measurements(path_curr)

        points_prev, points_curr, assoc = u.data_association(prev_frame,curr_frame)

        #Initial guess
        self.solver.initial_guess(self.cam, self.solver.map(), points_prev)

        #test(assoc, self.cam, self.solver.map(), points_curr)

        #picp
        self.picp(assoc)

        #test(assoc, self.cam, self.solver.map(), points_curr)
        
        self.prev_frame = curr_frame


def test(assoc, camera, world_points, point_curr):
    projected_points = []

    for _, idx_frame2 in assoc: 
        world_point = u.get_point(world_points, idx_frame2)
        if world_point is None: 
            continue

        predicted_image_point, is_valid =  camera.project_point(world_point)

        if not is_valid:
            continue
        else:
            projected_points.append((idx_frame2, predicted_image_point))

    visualize_projections_with_id(projected_points, point_curr)

def visualize_projections_with_id(projected_points, points_curr):
    x_proj, y_proj = [], []
    x_obs, y_obs = [], []

    projected_dict = {point[0]: point[1] for point in projected_points}
    observed_dict = {point[0]: point[1] for point in points_curr}

    common_ids = set(projected_dict.keys()).intersection(observed_dict.keys())
    for point_id in common_ids:
        proj = projected_dict[point_id]
        obs = observed_dict[point_id]

        x_proj.append(proj[0])
        y_proj.append(proj[1])
        x_obs.append(obs[0])
        y_obs.append(obs[1])

    if not x_proj or not x_obs:
        print("Nessun punto valido da visualizzare.")
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(x_obs, y_obs, c='red', label='Punti osservati', alpha=0.7, s=30)
    plt.scatter(x_proj, y_proj, c='blue', label='Punti proiettati', alpha=0.7, s=30)

    for i, point_id in enumerate(common_ids):
        plt.plot([x_obs[i], x_proj[i]], [y_obs[i], y_proj[i]], 'k--', linewidth=0.7, alpha=0.6)
        plt.text(x_obs[i], y_obs[i], str(point_id), fontsize=8, color='red')
        plt.text(x_proj[i], y_proj[i], str(point_id), fontsize=8, color='blue')

    plt.xlabel("Coordinata X (pixel)")
    plt.ylabel("Coordinata Y (pixel)")
    plt.title("Confronto tra punti osservati e proiettati (con ID e connessioni)")
    plt.legend()
    plt.grid(True)
    plt.show()
