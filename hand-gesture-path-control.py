import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# windows set up
from mediapipe.python.solutions.hands import HandLandmark

canvas = np.zeros((800,800,3), np.uint8)
canvas.fill(255)
g_x, g_y = int(0), int(0)
drawing = False
together = False

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

drawing_window = 'canvas'
cv2.namedWindow(drawing_window)
cv2.moveWindow(drawing_window, 0, 0)

camera_window = 'camera'
cv2.namedWindow(camera_window)
cv2.moveWindow(camera_window, 800, 0)

# refresh rate loop
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    canvas_height, canvas_width, _ = canvas.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            #drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                if id == HandLandmark.INDEX_FINGER_TIP:
                    if together:
                        cv2.circle(img=frame, center=(x,y), radius=15, color=(255, 0, 255))
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y
                if id == HandLandmark.MIDDLE_FINGER_TIP:
                    if together:
                        cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    middle_x = screen_width/frame_width*x
                    middle_y = screen_height/frame_height*y
                    if abs(index_x - middle_x) < 30:
                        together = True
                        pyautogui.moveTo(index_x, index_y)
                        if drawing and index_x<canvas_width and index_y<canvas_height:
                            g_x_temp, g_y_temp = int(index_x),int(index_y)
                            cv2.line(canvas, (int(g_x_temp), int(g_y_temp)), (int(g_x), int(g_y)), (255,0,0), thickness=3)
                            cv2.circle(img=frame, center=(x, y), radius=20, color=(255, 255, 255))
                            g_x, g_y = index_x,index_y
                    elif abs(index_x - middle_x) < 110:
                        pyautogui.moveTo(index_x, index_y)
                        g_x, g_y = int(index_x), int(index_y)
                        drawing = True
                        together = False
                    else:
                        g_x, g_y = int(index_x), int(index_y)
                        drawing = False
                        together = False
    cv2.imshow(drawing_window,canvas)
    cv2.imshow(camera_window, frame)
    # press 'esc' to exit loop
    if cv2.waitKey(1) & 0xFF == 27:
        break
