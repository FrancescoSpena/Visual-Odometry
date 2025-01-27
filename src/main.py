import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



if __name__ == "__main__":
    gt = u.read_traj()
    estimated_pose = []

    v = vo.VisualOdometry()
    status = v.init()
    T = v.cam.absolutePose()

    print(f"Status init: {status}")
    
    print(f"from frame {0} to frame {1}")
    print("-------------Estimation------------")
    print(f"T: {T}")
    print("-------------Ground Truth----------")
    T_gt_rel = u.relativeMotion(u.g2T(gt[0]),u.g2T(gt[1]))
    print(f"T: \n {T_gt_rel}")


    # iter = 2
    # for i in range(1, iter):
    #     v.run(i)
    #     T = v.cam.relativePose()

    #     print(f"from frame {i} to frame {i+1}")
    #     print("-------------Estimation------------")
    #     print(f"T: {T}")
    #     print("-------------Ground Truth----------")
    #     T_gt_rel = u.relativeMotion(u.g2T(gt[i]),u.g2T(gt[i+1]))
    #     print(f"T: \n {T_gt_rel}")


