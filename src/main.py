import numpy as np
from scipy.spatial.transform import Rotation as R
import VisualOdometry as vo
import utils as u
import rerun as rr

if __name__ == "__main__":
    v = vo.VisualOdometry()
    R, t, status = v.init()
    print("Init (with frame 0 and 1)")
    print(f"Status init: {status}")

    for i in range(1,10):
        R, t, status = v.run(i)
        print(f"from frame {i} to {i+1}")
        if status == True:
            print(f"R:\n {R}")
            print(f"t:\n {t}")
            print(f"delta:\n {v.dx}")
            print("========")
        else:
            print("No solution found.")
            print("========")
