import cv2
import mediapipe as mp
import numpy as np
import os
import time
path = "C:/users/akash/Desktop/SLT/"
data = [chr(i) for i in range(81, 65+26)]
writeit = False
counter = 501
try:
    os.mkdir(os.path.join(path, "Dataset"))
except:
    pass
for i in data:
    try:
        os.mkdir(os.path.join(path, "Dataset", i))
    except:
        pass
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands
hands = hands.Hands()
draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
while 1:
    ret, frame = cap.read()
    rimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rimg)
    temp = frame.copy()
    if res.multi_hand_landmarks:
        for l in res.multi_hand_landmarks:
            draw.draw_landmarks(temp, l,  mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
            h, w, c = temp.shape
            minX, minY, maxX, maxY = 500, 500, 0, 0
            for i in l.landmark:
                minX = min(minX, int(i.x * w)-50)  
                minY = min(minY, int(i.y * h)-50)
                maxX = max(maxX, int(i.x * w)+50)
                maxY = max(maxY, int(i.y * h)+50)
            cv2.rectangle(temp, (minX, minY),  (maxX, maxY), (0, 255, 0), 2)
    if writeit:
        if counter <= 1000:
            cv2.imwrite(os.path.join(path, "Dataset", data[0], str(counter)+".jpg"), cv2.cvtColor(frame[minY:maxY, minX:maxX], cv2.COLOR_BGR2GRAY))
            counter += 1
            print("Saving image: "+str(counter))
        else:
            writeit = False
            print("Done writing images to dataset") 
            counter = 0
    try:
        cv2.imshow(data[0], frame[minY:maxY, minX:maxX])
    except:
        pass
    cv2.imshow('frame', temp)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        writeit = True     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
