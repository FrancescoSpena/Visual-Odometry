import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

if __name__ == "__main__":
    v = vo.VisualOdometry()
    R, t, points_3d, status = v.init()

    print(f"Status: {status}")

    for i in range(1,120):
        v.run(i)