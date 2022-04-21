import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
ims = 100
train_img=[]
train_label=[]
test_img = []
j=0
labels_all = {"A" : 0, "B" : 1, "C": 2, "D" : 3, "E" : 4, "F" : 5, "G" : 6, "H" : 7, "I" : 8 }
path='train'
for i in os.listdir("train"):
    for j in os.listdir(os.path.join("train", i)):
        final_path=os.path.join(os.path.join('train', i, j))
        #print (j, final_path)
        img=cv2.imread(final_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(ims,ims))
        #img=img.astype('float32')
        train_img.append(img)
        train_label.append(labels_all[i])

train_img=np.array(train_img)
test_img=np.array(test_img)
train_label=np.array(train_label)
print(train_img.shape)
print(test_img.shape)
print(train_label.shape)

final_t = []
for i in range(len(train_img)):
  final_t.append([train_img[i], train_label[i]])
from random import shuffle
shuffle(final_t)

train_imgs = []
train_labels = []
for i in final_t:
  train_imgs.append(i[0])
  train_labels.append(i[1])
train_imgs=np.array(train_img)
train_labels=np.array(train_label)
print(train_img.shape)
print(train_label.shape)

model = Sequential()
# convolutional layer
model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(ims, ims, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
# output layer
model.add(Dropout(0.25))
model.add(Dense(26, activation='softmax'))

# compiling the sequential model
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(train_imgs, train_labels, batch_size=32, epochs=10)
model.save("/content/gdrive/My Drive/train.h5")

