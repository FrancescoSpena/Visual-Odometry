import numpy as np

def triangulate_n_views(point_tracks, projection_matrices):
    """
    Triangola un punto 3D da più viste usando il metodo DLT.

    Args:
        point_tracks (dict): { frame_id: (x, y) } punti 2D osservati in ogni frame
        projection_matrices (dict): { frame_id: P } matrici di proiezione 3x4 per ogni frame

    Returns:
        np.array: coordinate 3D del punto ricostruito [X, Y, Z]
    """
    A = []

    for frame_id, (x, y) in point_tracks.items():
        x, y = float(x), float(y)
        if frame_id not in projection_matrices:
            raise ValueError(f"Frame ID {frame_id} not found in projection_matrices.")
        
        P = projection_matrices[frame_id]  # Matrice di proiezione 3x4

        A.append(x * P[2] - P[0])  
        A.append(y * P[2] - P[1])  

    A = np.array(A)  
    

    _, _, V = np.linalg.svd(A)
    X = V[-1]  

    if X[3] == 0:
        raise ValueError("Homogeneous coordinate X[3] is zero. Cannot convert to Cartesian coordinates.")
    
    return X[:3] / X[3]


# Test script
if __name__ == "__main__":
    X_gt = np.array([1.0, 2.0, 3.0, 1.0])  # [X, Y, Z, W]
    X_gt2 = np.array([2.0, 4.0, 2.0, 1.0])  # [X, Y, Z, W]

    P0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])  # Identity projection (no translation or rotation)

    P1 = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])  # Identity projection (no translation or rotation)

    P2 = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])  # Translated projection (shifted along X-axis)

    P3 = np.array([
        [1, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    P4 = np.array([
        [1, 0.1, 0, 2],
        [0.1, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    
    x0 = P0 @ X_gt
    x1 = P1 @ X_gt  
    x2 = P2 @ X_gt  
    x3 = P3 @ X_gt
    x4 = P4 @ X_gt

    x0 = x0[:2] / x0[2]
    x1 = x1[:2] / x1[2]
    x2 = x2[:2] / x2[2]
    x3 = x3[:2] / x3[2]
    x4 = x4[:2] / x4[2]

    point_tracks = {
        0: x0,
        1: x1,  
        2: x2,   
        3: x3,
        4: x4
    }

    projection_matrices = {
        0: P0,
        1: P1,  
        2: P2,   
        3: P3,
        4: P4
    }

    X_reconstructed = triangulate_n_views(point_tracks, projection_matrices)

    print("Ground truth 3D point:", X_gt[:3])
    print("Reconstructed 3D point:", X_reconstructed)
    print("Error (L2 norm):", np.round(np.linalg.norm(X_gt[:3] - X_reconstructed)))

    x1 = P1 @ X_gt2  
    x2 = P2 @ X_gt2 
    x3 = P3 @ X_gt2
    x4 = P4 @ X_gt2

    x1 = x1[:2] / x1[2]
    x2 = x2[:2] / x2[2]
    x3 = x3[:2] / x3[2]
    x4 = x4[:2] / x4[2]

    point_tracks = {
        1: x1,  
        2: x2,   
        3: x3,
        4: x4
    }

    projection_matrices = {
        1: P1,  # View 1
        2: P2,   # View 2
        3: P3,
        4: P4
    }

    print("Ground truth 3D point:", X_gt[:3])
    print("Reconstructed 3D point:", X_reconstructed)
    print("Error (L2 norm):", np.round(np.linalg.norm(X_gt[:3] - X_reconstructed)))