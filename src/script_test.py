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
        if frame_id not in projection_matrices:
            raise ValueError(f"Frame ID {frame_id} not found in projection_matrices.")
        
        P = projection_matrices[frame_id]  # Matrice di proiezione 3x4

        # Creiamo due equazioni lineari per ogni vista
        A.append(x * P[2] - P[0])  # Prima equazione
        A.append(y * P[2] - P[1])  # Seconda equazione

    A = np.array(A)  # Converte in una matrice (2N x 4)
    
    # Risolviamo il sistema AX = 0 usando SVD
    _, _, V = np.linalg.svd(A)
    X = V[-1]  # L'ultima riga di V è la soluzione del sistema
    
    # Convertiamo in coordinate cartesiane dividendo per W
    if X[3] == 0:
        raise ValueError("Homogeneous coordinate X[3] is zero. Cannot convert to Cartesian coordinates.")
    
    return X[:3] / X[3]


# Test script
if __name__ == "__main__":
    # Ground truth 3D point (in homogeneous coordinates)
    X_gt = np.array([1.0, 2.0, 3.0, 1.0])  # [X, Y, Z, W]

    # Define projection matrices for two views (3x4 matrices)
    P1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])  # Identity projection (no translation or rotation)

    P2 = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])  # Translated projection (shifted along X-axis)

    # Project the 3D point into 2D for each view
    x1 = P1 @ X_gt  # Projection in view 1
    x2 = P2 @ X_gt  # Projection in view 2

    # Normalize to get 2D coordinates (divide by Z)
    x1 = x1[:2] / x1[2]
    x2 = x2[:2] / x2[2]

    # Create point_tracks and projection_matrices dictionaries
    point_tracks = {
        1: x1,  # View 1
        2: x2   # View 2
    }

    projection_matrices = {
        1: P1,  # View 1
        2: P2   # View 2
    }

    # Reconstruct the 3D point using the function
    X_reconstructed = triangulate_n_views(point_tracks, projection_matrices)

    # Compare with ground truth
    print("Ground truth 3D point:", X_gt[:3])
    print("Reconstructed 3D point:", X_reconstructed)
    print("Error (L2 norm):", np.round(np.linalg.norm(X_gt[:3] - X_reconstructed)))