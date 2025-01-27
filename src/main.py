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
    T_gt_rel = u.g2T(gt[0])
    print(f"T: \n {T_gt_rel}")
    print("\n")

    '''
    gt[0] represent the (x, y, theta) from frame 0 to 1
    gt[1] represent the (x, y, theta) from frame 0 to 2
    
    if: 

    0_T_1 = u.g2T(gt[0]) is the transformation between the frame 0 and 1
    0_T_2 = u.g2T(gt[1]) is the transformation between the frame 0 and 2

    1_T_2 = 1_T_0 @ 0_T_2 = inv(0_T_1) @ 0_T_2 

    1_T_2 is the relative transformation between frame 1 and 2
    '''

    'from frame 1 to 2'
    'idx=index of the next frame'
    v.run(2)
    #Relative pose (from frame 1 to frame 2)
    T = v.cam.relativePose()

    print(f"from frame {1} to frame {2}")
    print("-------------Estimation------------")
    print(f"T: {T}")
    print("-------------Ground Truth----------")
    T01 = u.g2T(gt[0])
    T02 = u.g2T(gt[1])
    T_gt_rel = u.relativeMotion(T01, T02)
    print(f"T: \n {T_gt_rel}")
    print("\n")




