import numpy as np

def align_pose(T_est, T_gt):
    """
    Aligns the estimated transformation matrix to the ground truth frame.
    
    Parameters:
    - T_est: (4x4) Estimated transformation matrix
    - T_gt: (4x4) Ground truth transformation matrix
    
    Returns:
    - T_corrected: (4x4) Corrected transformation matrix
    """
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

    # Compute the correction rotation: R_corr * R_est = R_gt
    R_corr = R_gt @ np.linalg.inv(R_est)

    # Correct the estimated transformation
    T_corrected = np.eye(4)
    T_corrected[:3, :3] = R_corr @ R_est  # Apply rotation correction
    T_corrected[:3, 3] = R_corr @ t_est  # Rotate translation vector

    return T_corrected

# Given data
T_est = np.array([
    [0.99991196, -0.00146693,  0.01318819,  0.29789704],
    [0.00136473,  0.999969,    0.007755,    0.15522672],
    [-0.01319916, -0.00773631, 0.99988296, -0.94189278],
    [0., 0., 0., 1.]
])

T_gt = np.array([
    [1.0, 0.0, 0.0, 0.200426],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Apply transformation alignment
T_corrected = align_pose(T_est, T_gt)

# Display results
print("Corrected Transformation:")
print(T_corrected)
