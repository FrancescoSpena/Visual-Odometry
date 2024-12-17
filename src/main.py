from VisualOdometry import VisualOdometry
import numpy as np
import utils as u

if __name__ == '__main__':
    vo = VisualOdometry()
    gt = vo.gt
    #homogeneous transformation from i to i+1
    T_prev = None
    T_curr = None
    T_gt_prev = None
    T_gt_curr = None

    R = None
    t = None
    for i in range(0,10):
        if(i == 0):
            T_prev = vo.run(i)
            T_curr = T_prev
            T_gt_prev = u.gt2T(gt[i])
            T_gt_curr = T_gt_prev
        else:
            T_curr = vo.run(i)
            T_gt_curr = u.gt2T(gt[i])
        
            rel_T = np.linalg.inv(T_prev) @ T_curr
            rel_gt = np.linalg.inv(T_gt_prev) @ T_gt_curr

            error_T = np.linalg.inv(rel_T) @ rel_gt
            
            R = np.trace(np.eye(3) - error_T[:3, :3])
            
            norm_rel_T = np.linalg.norm(rel_T[:3, 3])
            norm_rel_GT = np.linalg.norm(rel_gt[:3, 3])

            translation_ratio = norm_rel_T / norm_rel_GT

            print(f"t = {translation_ratio}")

            T_prev = T_curr
            T_gt_prev = T_gt_curr