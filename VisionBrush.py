# importing the libraries
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# the color codes are embedded into a list
# red, blue, green and yellow are the chose colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
# the color index is set to 0
colorIndex = 0

# the dequeues and the indices are created for the colors
blue_points = [deque(maxlen=1024)]
green_points = [deque(maxlen=1024)]
red_points = [deque(maxlen=1024)]
yellow_points = [deque(maxlen=1024)]

# indexes are created for each color to mark the points
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# a kernel is initialized as a 5 x 5 matrix of 1's
# the kernel is used for noise reduction and to enhance the features detected
# it also helps to improve contour detection, making the edges more prominent
kernel = np.ones((5, 5), np.uint8)