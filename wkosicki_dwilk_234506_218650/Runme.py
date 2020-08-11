from keras.models import load_model
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plot

def grayscale(picture):
    res = Image.new(picture.mode, picture.size)
    width, height = picture.size
    for i in range(0, width):
        for j in range(0, height):
            pixel = picture.getpixel((i, j))
            avg = (pixel[0] + pixel[1] + pixel[2]) / 3
            res.putpixel((i, j), (int(avg), int(avg), int(avg)))
    # res.show()
    return res

def normalize(picture):
    width, height = picture.size
    normalized_array = []
    for j in range(0, height):
        for i in range(0, width):
            pixel = picture.getpixel((i, j))
            normalized_array.append(pixel[0] / 255.0)
    return np.array(normalized_array)

if __name__ == "__main__":
    recognizedCars = 0
    totalNumNumCars = 0

    model = load_model('carDetectionCNN.h5')
    row, column = 64, 64
    path = 'test/'

    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
    else:
        filename = path

    filesToCheck = os.listdir(filename)
    for file in filesToCheck:
        # print("SPRAWDZAM " + str(file))
        img = Image.open(path + file)
        # print("BITS " + str(img.bits))
        img = img.resize((row, column), Image.ANTIALIAS)
        gray_image = grayscale(img)
        X_test = normalize(gray_image)
        X_test = X_test.reshape(1, row, column, 1)
        classes = model.predict(X_test)
        maxVal = classes[0].max()
        indexVal = np.where(classes[0] == maxVal)
        totalNumNumCars = totalNumNumCars + 1
        if indexVal[0] == 0:
            print("NA OBRAZIE " + str(file) + " ZNALEZIONO SAMOCHOD")
            recognizedCars = recognizedCars + 1
        else:
            print("NA OBRAZIE " + str(file) + " NIE ZNALEZIONO SAMOCHODU")

print("\nZNALEZIONO " + str(recognizedCars) + " SAMOCHODOW\n")

# ratio = round((recognizedCars / totalNumNumCars * 100), 2)
# print("Rozpoznano " + str(ratio) + "% samochodow")
