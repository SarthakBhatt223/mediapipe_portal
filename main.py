import cv2

import numpy as np
import mediapipe as mp
import math
import time

#creating objects

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

#webcam initilization

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


#mp hands draw 
if not cap.isOpened():
    print("webcam not ffound")
    exit()


with mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5,
) as hands:
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        thumbs_extended = 0
        index_extended = 0

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                label = results.multi_handedness[idx].classification[0].label

                # Thumb opens sideways, so we compare X coordinates.
                if label == "Right":
                    thumb_open = lm[4].x > lm[3].x
                else:
                    thumb_open = lm[4].x < lm[3].x

                # Index opens upward, so we compare Y coordinates.
                index_open = lm[8].y < lm[6].y and lm[8].y < lm[5].y

                if thumb_open:
                    thumbs_extended += 1
                if index_open:
                    index_extended += 1

            if thumbs_extended == 2 and index_extended == 2:
                print("Portal detected")
            

        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




