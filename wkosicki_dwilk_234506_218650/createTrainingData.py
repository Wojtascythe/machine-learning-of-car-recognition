# import classes and functions
from PIL import Image
import matplotlib.pyplot as plot
import os
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Sciezki do kolorowych obrazow
pathTrainDataCar = 'TrainImages/train/car/'
pathTrainDataOther = 'TrainImages/train/other/'
pathTestDataCar = 'TrainImages/validation/car/'
pathTestDataOther = 'TrainImages/validation/other/'

# Sciezki do zmniejszonych obrazow w skali szarosci
pathResizedTrainDataCar = 'ResizeTrainImages/train/car/'
pathResizedTrainDataOther = 'ResizeTrainImages/train/other/'
pathResizedTestDataCar = 'ResizeTrainImages/validation/car/'
pathResizedTestDataOther = 'ResizeTrainImages/validation/other/'

# Rozmiar zdjec
row, column = 64, 64

listingCar = os.listdir(pathTrainDataCar)
print(listingCar)
for file in listingCar:
	img = Image.open(pathTrainDataCar + file)
	resizeImg = img.resize((row, column))
	gray = resizeImg.convert('L')
	gray.save(pathResizedTrainDataCar + file)

listingOther = os.listdir(pathTrainDataOther)
print(listingOther)
for file in listingOther:
	img = Image.open(pathTrainDataOther + file)
	resizeImg = img.resize((row, column))
	gray = resizeImg.convert('L')
	gray.save(pathResizedTrainDataOther + file)

listingTestCar = os.listdir(pathTestDataCar)
print(listingTestCar)
for file in listingTestCar:
	img = Image.open(pathTestDataCar + file)
	resizeImg = img.resize((row, column))
	gray = resizeImg.convert('L')
	gray.save(pathResizedTestDataCar + file)

listingTestRandom = os.listdir(pathTestDataOther)
print(listingTestRandom)
for file in listingTestRandom:
	img = Image.open(pathTestDataOther + file)
	resizeImg = img.resize((row, column))
	gray = resizeImg.convert('L')
	gray.save(pathResizedTestDataOther + file)




