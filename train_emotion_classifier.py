from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import Callback
from keras.initializers import TruncatedNormal
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import keras
import subprocess
import os

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.batch_size = 32
config.num_epochs = 20

input_shape = (48, 48, 1)

def load_fer2013():
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output("curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

model = Sequential()

#model.add(Conv2D(64,
#    (3, 3),
#    input_shape=(48, 48,1),
#    activation='relu'))

#model.add(Conv2D(64,
#    (3, 3),strides=(2, 2), padding="SAME",
#    input_shape=(48, 48,1),
#    activation='relu', use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001)))

#model.add(Dropout(0.5, input_shape=(48, 48,1)))

#model.add(Conv2D(32, (5,5), padding="same", activation='relu', input_shape=(width, height, channels) ))


model.add(Conv2D(32,
    (3, 3), 
    padding="same",
    input_shape=(48, 48,1),
    activation='relu'))

model.add(Dropout(0.5))
#model.add(Conv2D(32,
 #   (3, 3),
 #   input_shape=(48, 48,1),
 #   activation='relu'))
model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#model.add(Conv2D(32,
#    (3, 3),
#    input_shape=(48, 48,1),
#    activation='relu'))

#model.add(Conv2D(32,
#    (3, 3),
#    input_shape=(48, 48,1),
#    activation='relu'))

model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3,3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Conv2D(32, 4, 4,input_shape=(48, 48,1), activation='relu'))

model.add(Flatten(input_shape=input_shape))
#model.add(Dense(num_classes, activation="softmax"))

model.add(Dense(num_classes, activation='relu'))
#model.add(Dense(units = 128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
#model.add(Dense(units = 1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
metrics=['accuracy'])

model.fit(train_faces, train_emotions, batch_size=config.batch_size,
        epochs=config.num_epochs, verbose=1, callbacks=[
            WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        ], validation_data=(val_faces, val_emotions))


model.save("emotion.h5")



