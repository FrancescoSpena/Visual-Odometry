import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

if __name__ == "__main__":
    v = vo.VisualOdometry()
    R, t, _, status = v.init()

    print(f"Status init: {status}")
    print("Estimation...")
    for i in range(1,120):
        R, t = v.run(i)
    
    print("Estimation complete.")