import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

if __name__ == "__main__":
    v = vo.VisualOdometry()
    R, t, status = v.init()
    print(f"Status init: {status}")

    for i in range(1,10):
        R, t = v.run(i)
        print(f"from frame {i} to {i+1}")
        print(f"R:\n {R}")
        print(f"t:\n {t}")
        print("========")

