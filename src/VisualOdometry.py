import utils as u
import numpy as np
import PICP_solver as solver
import Camera as camera
import cv2
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

        other_info_frame_0 = u.extract_other_info(path0)
        other_info_frame_1 = u.extract_other_info(path1)

        self.prev_frame = data_frame_0
        
        #points0: frame 0, points1: frame1 --> (ID, (X,Y))
        points0, points1, assoc = u.data_association(data_frame_0, 
                                                     data_frame_1)
        
        p_0 = np.array([item[1] for item in points0])
        p_1 = np.array([item[1] for item in points1])

        gt_0 = np.array(other_info_frame_0['Ground_Truths'])
        gt_1 = np.array(other_info_frame_1['Ground_Truths'])

        gt_dist = np.linalg.norm(gt_1 - gt_0)

        #Pose from 0 to 1
        R, t = u.compute_pose(p_0,
                              p_1,
                              self.cam.K,
                              gt_dist)
        
        #3D points of the frame 0
        points_3d = u.triangulate(R,
                                  t,
                                  p_0,
                                  p_1,
                                  self.cam.K,
                                  assoc)
        
    
        self.cam.update_absolute(u.m2T(R, t))
        self.cam.update_relative(u.m2T(R, t))

        self.solver.set_map(points_3d)
        self.solver.set_image_points(points0)

        #Check
        if(np.linalg.det(R) != 1 or np.linalg.norm(t) == 0):
            self.status = False
        
        return self.status
    
    def run(self, idx):
        'Update pose'
        path_curr = u.generate_path(idx)
        data_curr = u.extract_measurements(path_curr)

        point_prev, points_curr, assoc = u.data_association(self.prev_frame,data_curr)

        world_points = self.solver.map()

        self.solver.initial_guess(self.cam, world_points, point_prev)

        test(assoc, self.cam, world_points, points_curr)
        for _ in range(50):
            self.solver.one_round(assoc)
            dx_norm = np.linalg.norm(self.solver.dx)
            self.cam.update_pose(self.solver.dx)
            print(f"Variazione posa (dx): {dx_norm:.6f}")

            if dx_norm <= 0.2:
                break
        
        test(assoc, self.cam, world_points, points_curr)

        self.prev_frame = data_curr
        self.solver.set_image_points(points_curr)


def test(assoc, camera, world_points, point_curr):
    projected_points = []

    for _, idx_frame2 in assoc: 
        world_point = u.get_point(world_points, idx_frame2)
        if world_point is None: 
            continue

        world_point_h = np.append(world_point, 1)
        point_in_camera = camera.absolute_pose() @ world_point_h
        point_in_camera = point_in_camera[:3] / point_in_camera[3]

        predicted_image_point, is_valid = u.project_point(point_in_camera, camera.K)

        if not is_valid:
            continue
        else:
            projected_points.append((idx_frame2, predicted_image_point))

    visualize_projections(projected_points, point_curr)


def visualize_projections(projected_points, points_curr):
    """
    Visualizza i punti proiettati e osservati nel piano immagine.

    Args:
        projected_points: Lista di punti proiettati nel formato (id, array([x_proj, y_proj])).
        points_curr: Lista di feature osservate nel formato (id, array([x_obs, y_obs])).

    Returns:
        None
    """
    # Crea liste di coordinate per i punti osservati e proiettati
    x_proj, y_proj = [], []
    x_obs, y_obs = [], []

    # Crea dizionari per un accesso più semplice tramite id
    projected_dict = {point[0]: point[1] for point in projected_points}
    observed_dict = {point[0]: point[1] for point in points_curr}

    # Trova gli id comuni e raccogli le coordinate corrispondenti
    common_ids = set(projected_dict.keys()).intersection(observed_dict.keys())
    for point_id in common_ids:
        proj = projected_dict[point_id]
        obs = observed_dict[point_id]

        x_proj.append(proj[0])
        y_proj.append(proj[1])
        x_obs.append(obs[0])
        y_obs.append(obs[1])

    # Verifica che ci siano punti da visualizzare
    if not x_proj or not x_obs:
        print("Nessun punto valido da visualizzare.")
        return

    # Plotta i punti osservati
    plt.scatter(x_obs, y_obs, c='red', label='Punti osservati', alpha=0.7, s=30)

    # Plotta i punti proiettati
    plt.scatter(x_proj, y_proj, c='blue', label='Punti proiettati', alpha=0.7, s=30)

    # Aggiungi etichette e legenda
    plt.xlabel("Coordinata X (pixel)")
    plt.ylabel("Coordinata Y (pixel)")
    plt.title("Confronto tra punti osservati e proiettati")
    plt.legend()
    plt.grid(True)
    plt.show()
