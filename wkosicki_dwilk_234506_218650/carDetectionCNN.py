from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D,Dropout, Flatten
from keras.utils import np_utils
from PIL import Image
import os,sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from array import array
import numpy as np 
import matplotlib.pyplot as plot
np.random.seed(0)
row, column = 64, 64

pathResizedTrainDataCar = 'ResizeTrainImages/train/car/'
pathResizedTrainDataOther = 'ResizeTrainImages/train/other/'
pathResizedTestDataCar = 'ResizeTrainImages/validation/car/'
pathResizedTestDataOther = 'ResizeTrainImages/validation/other/'

X_train = []
Y_train = []
X_test = []
Y_test = []
classes = 2 

# SAMOCHOD
listingCar = os.listdir(pathResizedTrainDataCar)
for file in listingCar:
	img = Image.open(pathResizedTrainDataCar + file)
	x = img_to_array(img)
	X_train.append(x)
	Y_train.append(0)

# INNE OBRAZY
listingRandom = os.listdir(pathResizedTrainDataOther)
for file in listingRandom:
	img = Image.open(pathResizedTrainDataOther + file)
	x = img_to_array(img)
	X_train.append(x)
	Y_train.append(1)

# TEST
listingTestData = os.listdir(pathResizedTestDataCar)
for file in listingTestData:
	img = Image.open(pathResizedTestDataCar + file)
	x = img_to_array(img)
	X_test.append(x)
	Y_test.append(0)

listingTestData = os.listdir(pathResizedTestDataOther)
for file in listingTestData:
	img = Image.open(pathResizedTestDataOther + file)
	x = img_to_array(img)
	X_test.append(x)
	Y_test.append(1)

total_input = len(X_train)

X_train = np.array(X_train)
X_train = X_train.reshape(total_input, row, column, 1) 
X_train = X_train.astype('float32')     
X_train /= 255 
Y_train = np.array(Y_train)   
Y_train = Y_train.reshape(total_input, 1)   

print("X_train:")
print(X_train.shape)
print("Y_train:")
print(Y_train.shape)

total_testData = len(X_test)
print("Total Test Data : %d" %total_testData)

X_test = np.array(X_test)
X_test = X_test.reshape(total_testData, row, column, 1)
X_test = X_test.astype('float32')     
X_test /= 255 
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(total_testData, 1)  

# 0 - samochod, 1 - cos innego
print(Y_train.shape)
print(Y_train[0])

Y_train = np_utils.to_categorical(Y_train, classes) 
Y_test = np_utils.to_categorical(Y_test, classes)

# Ustalenie parametrow
input_size = row * column
batch_size = 32 # TESTOWANO 32
hidden_neurons = 30
epochs = 10

# Stworzenie modelu
model = Sequential() 
model.add(Convolution2D(32, (2, 2), input_shape=(row, column, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(hidden_neurons)) 
model.add(Activation('relu'))      
model.add(Dense(classes)) 
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adadelta')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)
print('\n''Test accuracy:', score[1]) 

# Zapis klasyfikatora
model.save('carDetectionCNN.h5')






