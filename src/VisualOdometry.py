import utils as u
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualOdometry():
    def __init__(self, camera_path='../data/camera.dat'):
        self.camera_info = u.extract_camera_data(camera_path)
        self.K = self.camera_info['camera_matrix']
        self.width = self.camera_info['width']
        self.height = self.camera_info['height']
        self.z_near = self.camera_info['z_near']
        self.z_far = self.camera_info['z_far']

    def init(self):
        pass
        