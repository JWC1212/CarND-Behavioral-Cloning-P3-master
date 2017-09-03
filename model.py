# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import csv
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

lines = []
with open('../data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

lines = shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def sample_generator (samples, batch_size=32):
    n_samples = len(samples)
    shuffle(samples)
    while 1:
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]            
            images, measurements = [], []
            for line in batch_samples:
                src_file_path = line[0]
                file_name = src_file_path.split('/')[-1]
                current_path = '../data/IMG/'+file_name
                img = cv2.imread(current_path)
                images.append(img)
                measurement = float(line[3])
                measurements.append(measurement)

            augmented_images, augmented_measurements = [],[]
            for img, measure in zip(images, measurements):
                augmented_images.append(img)
                augmented_measurements.append(measure)
                augmented_images.append(cv2.flip(img, 1))
                augmented_measurements.append(measure*(-1.0))

            X_ = np.array(augmented_images)
            y_ = np.array(augmented_measurements)
            yield shuffle(X_, y_)

train_generator = sample_generator(train_samples, 128)
validate_generator = sample_generator(validation_samples, 128)
    
from keras.models import Sequential
from keras.layers.core import Flatten, Dense ,Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(6, (5, 5),strides=(1, 1), padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(6, (5, 5),strides=(1, 1), padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), epochs=10, verbose=1, validation_data=validate_generator,validation_steps=len(validation_samples))
model.save('model.h5')

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()