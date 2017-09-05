# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import csv
import matplotlib.image as mpimg
from sklearn.utils import shuffle

lines = []
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

lines = shuffle(lines)
n_lines = len(lines)
n_train = int(0.8*n_lines)
train_samples, validation_samples = lines[0:n_train], lines[n_train:]

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
                current_path = './data/IMG/'+file_name
                img = mpimg.imread(current_path)
                images.append(np.flip(img,1))
                measurement = float(line[3])
                measurements.append(measurement*(-1.0))
            X_ = np.array(images)
            y_ = np.array(measurements)
            yield shuffle(X_, y_)

train_generator = sample_generator(train_samples, batch_size=128)
validate_generator = sample_generator(validation_samples, batch_size=128)
    
from keras.models import Sequential
from keras.layers import Flatten, Dense ,Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, activation='relu'))
#model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
model.add(Convolution2D(36, 5, 5, activation='relu'))
#model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))
#model.add(Convolution2D(48, 5, 5, activation='relu'))
#model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
#model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), nb_epoch=3, verbose=1, validation_data=validate_generator,nb_val_samples=len(validation_samples))
model.save('model.h5')
'''
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''
