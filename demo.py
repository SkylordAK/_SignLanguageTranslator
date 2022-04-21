import os
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical
from random import shuffle
import numpy as np
import cv2
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
ALL_LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()
#ALL_LABELS = 'A B C D x E F G H I J K L M N x O P Q R S x T U V W X Y Z'.split()
IMG_SIZE = 125
LR = 0.001
#MODEL_NAME = 'SignLanguage_DNN_{}_{}x{}_conv.model'.format(LR, IMG_SIZE, IMG_SIZE)
MODEL_NAME = 'SignLanguage_DNN_{}_{}x{}_conv.model'.format(LR, IMG_SIZE, IMG_SIZE)
KERNEL_SIZE = 5
BATCH_1 = 32
BATCH_2 = 64
TRAIN_DIR = 'Dataset/'
ims = 100
frame_rate = 0
text = ''
cap = cv2.VideoCapture(0)
hands = mp.solutions.hands
hands = hands.Hands()
draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

model = keras.models.load_model("train_3.h5")
while 1:
    frame_rate += 1
    ret, frame = cap.read()

    word_space = np.array([[[255]*100]*500]*3, np.uint8).reshape(100, 500, 3)
    rimg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rimg)
    temp = frame.copy()
    alpha = 1.0
    beta = 0.2
    if res.multi_hand_landmarks:
        for l in res.multi_hand_landmarks:
            draw.draw_landmarks(temp, l,  mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
            h, w, c = temp.shape
            minX, minY, maxX, maxY = 500, 500, 0, 0
            dis = 40
            for i in l.landmark:
                minX = min(minX, int(i.x * w)-dis)  
                minY = min(minY, int(i.y * h)-dis)
                maxX = max(maxX, int(i.x * w)+dis)
                maxY = max(maxY, int(i.y * h)+dis)
            crop_img = frame[minY:maxY, minX:maxX] 

            cv2.rectangle(temp, (minX, minY),  (maxX, maxY), (0, 255, 0), 2)
            try:
                pimg = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            except:
                pass
            if frame_rate % 20 == 0:
                if max(model.predict([pimg])[0]) > 0.0:
                    text += 'W'#ALL_LABELS[np.argmax(model.predict([pimg])[0])]
                else:
                    text = ''
                print (ALL_LABELS[np.argmax(model.predict([pimg])[0])])
                frame_rate = 0
    cv2.putText(word_space, text.upper(), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('prediction', word_space)
    cv2.imshow('frame', temp)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
