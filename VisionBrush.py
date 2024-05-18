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

# the canvas is created on a white background
paint_box = np.zeros((471, 610, 3)) + 255
paint_box = cv2.rectangle(paint_box, (40, 1), (140, 60), (0, 0, 0), 2)
paint_box = cv2.rectangle(paint_box, (160, 1), (250, 60), (255, 0, 0), 2)
paint_box = cv2.rectangle(paint_box, (270, 1), (360, 60), (0, 255, 0), 2)
paint_box = cv2.rectangle(paint_box, (380, 1), (470, 60), (0, 0, 255), 2)
paint_box = cv2.rectangle(paint_box, (490, 1), (580, 60), (0, 255, 255), 2)

# the rectangles are filled with the respective colors
cv2.rectangle(paint_box, (490, 1), (580, 60), (255, 255, 0), -1)
cv2.rectangle(paint_box, (160, 1), (250, 60), (255, 0, 0), -1)  # rectangle is folled with blue
cv2.rectangle(paint_box, (270, 1), (360, 60), (0, 255, 0), -1)  # rectangle is filled with green
cv2.rectangle(paint_box, (380, 1), (470, 60), (0, 0, 255), -1)  # rectangle is filled with red
cv2.rectangle(paint_box, (490, 1), (580, 60), (0, 255, 255), -1)  # rectangle is filled with yellow

# the text is added to the canvas
cv2.putText(paint_box, "CLEAR", (49,33), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paint_box, "BLUE", (185, 33), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_box, "GREEN", (298, 33), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_box, "RED", (420, 33), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paint_box, "YELLOW", (520, 33), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# creating an instance of mediapipe
media_pipe_hands = mp.solutions.hands

# the number of hands are specified and the confidence level of detection is set to 0.7
hands = media_pipe_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# the drawing utilities are imported
media_pipe_Draw = mp.solutions.drawing_utils

# the web camera is intiialized 
capture_device = cv2.VideoCapture(0)

# a boolean object is created and set to True
bool = True

# the loop runs until the webcam is on
while bool:
    # each frame in the webcam is read
    bool, screen_frame = capture_device.read()

    # the dimensions of the frame are stored in x, y and z
    x, y, z = screen_frame.shape

    # the frame is flipped vertically
    screen_frame = cv2.flip(screen_frame, 1)
    
    # the frame is converted to RGB
    screen_frame_rgb = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2RGB)

    # the rectangles are drawn on the frame
    screen_frame = cv2.rectangle(screen_frame, (40, 1), (140, 65), (0, 0, 0), 2)
    screen_frame = cv2.rectangle(screen_frame, (160, 1), (250, 65), (255, 0, 0), 2)
    screen_frame = cv2.rectangle(screen_frame, (270, 1), (360, 65), (0, 255, 0), 2)
    screen_frame = cv2.rectangle(screen_frame, (380, 1), (470, 65), (0, 0, 255), 2)
    screen_frame = cv2.rectangle(screen_frame, (490, 1), (580, 65), (0, 255, 255), 2)

    # the text is added to the frame
    cv2.putText(screen_frame, "CLEAR", (49,33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) # text for clear
    cv2.putText(screen_frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) # text for blue
    cv2.putText(screen_frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) # text for green
    cv2.putText(screen_frame, "RED", (420, 33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) # text for red
    cv2.putText(screen_frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA) # text for yellow

    # obtaining the results from the hands
    hand_points = hands.process(screen_frame_rgb)

    # processing the results
    if hand_points.multi_hand_landmarks:
        landmarks = []

        # the landmarks are stored in the handslms
        for handslms in hand_points.multi_hand_landmarks:
            for lm in handslms.landmark:
                # printing the x and y coordinates
                print(lm.x)
                print(lm.y)

                # the x and y coordinates are stored in lmx and lmy
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append((lmx, lmy))

        # the landmarks are drawn on the frame
        media_pipe_Draw.draw_landmarks(screen_frame, handslms, media_pipe_hands.HAND_CONNECTIONS)

    # the forefinger position is marked
    fore_finger = (landmarks[8][0], landmarks[8][1])

    # the center is set to the forefinger
    center = fore_finger

    # the thumb position is marked
    thumb = (landmarks[4][0], landmarks[4][1])

    # a circle is drawn at the center
    cv2.circle(screen_frame, center, 3, (0, 255, 0), -1)

    # difference between y- coordinates = y- coordinate of thumb - y-coordinate of center
    difference = center[1] - thumb[1]
    print(difference)

    # for each difference, if it is less than 30, points to be appended to the dequeues of each color
    if (thumb[1] - center[1] < 30):
        # the blue points are appended to the dequeues
        blue_points.append(deque(maxlen=512))
        blue_index = blue_index + 1

        # the green points are appended to the dequeues
        green_points.append(deque(maxlen=512))
        green_index = green_index + 1

        # the red points are appended to the dequeues
        red_points.append(deque(maxlen=512))
        red_index += 1

        # the yellow points are appended to the dequeues
        yellow_points.append(deque(maxlen=512))
        yellow_index += 1
    
    # if the center is less than or equal to 65
    elif center[1] <= 65:
        # if the center is between 40 and 140, the canvas is cleared
        if 40 <= center[0] <=140:
            # the blue points are set to dequeues
            blue_points = [deque(maxlen=512)]
            # the green points are set to dequeues
            green_points = [deque(maxlen=512)]
            # the red points are set to dequeues
            red_points = [deque(maxlen=512)]
            # the yellow points are set to dequeues
            yellow_points = [deque(maxlen=512)]

            # the indexes are set to zero
            blue_index = 0
            green_index = 0
            red_index = 0
            yellow_index = 0

            # the paint window is set to white
            paint_box[67:, :, :] = 255

        # if the center is between 160 and 255, the color index is set to 0
        elif 160 <= center[0] <= 250:
            colorIndex = 0 # blue color

        # if the center is between 270 and 360, the color index is set to 1
        elif 270 <= center[0] <= 360:
            colorIndex = 1 # green color

        # if the center is between 380 and 470, the color index is set to 2
        elif 380 <= center[0] <= 470:
            colorIndex = 2 # red color

        # if the center is between 490 and 580, the color index is set to 3
        elif 490 <= center[0] <= 580:
            colorIndex = 3 # yellow color
    
        else: 
            # if the color index is 0, the blue points are appended to the dequeues
            if colorIndex == 0:
                blue_points[blue_index].appendleft(center)

            # if the color index is 1, the green points are appended to the dequeues
            elif colorIndex == 1:
                green_points[green_index].appendleft(center)

            # if the color index is 2, the red points are appended to the dequeues
            elif colorIndex == 2:
                red_points[red_index].appendleft(center)

            # if the color index is 3, the yellow points are appended to the dequeues
            elif colorIndex == 3:
                yellow_points[yellow_index].appendleft(center)

    # the next dequeues are appended when nothing is detected
    else:
        # the blue points are appended to the dequeues
        blue_points.append(deque(maxlen=512))
        blue_index += 1
    
        # the green points are appended to the dequeues
        green_points.append(deque(maxlen=512))
        green_index += 1
    
        # the red points are appended to the dequeues
        red_points.append(deque(maxlen=512))
        red_index += 1
    
        # the yellow points are appended to the dequeues
        yellow_points.append(deque(maxlen=512))
        yellow_index += 1
    
    # the lines of all the colors are drawn on the canvas and the frame
    points = [blue_points, green_points, red_points, yellow_points]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(screen_frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paint_box, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # the frame is displayed
    cv2.imshow("Frame", screen_frame)

    # the canvas is displayed
    cv2.imshow("Canvas", paint_box)

    # if the key 'z' is pressed, the loop breaks
    if cv2.waitKey(1) == ord('z'):
        break

# the webcam is stopeed 
capture_device.release()

# all the windows are closed
cv2.destroyAllWindows()