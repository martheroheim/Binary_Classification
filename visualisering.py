import numpy as np
import matplotlib.pyplot as plt
from loader import get_dataset



Y0,c=get_dataset(dataset="training", digit1=2, digit2=7, path=".")
c = c
Y0 = Y0


amount = 4
lines = 2
columns = 2
image = np.zeros((10000, 28, 28))
number = np.zeros(amount)

def printPictures(fra, amount):
    for i in range(fra, fra + amount):
        image[i-fra] = Y0[:,i].reshape(28, 28)

    fig = plt.figure()

    for i in range(amount):
        ax = fig.add_subplot(lines, columns,  1 + i)
        plt.imshow(image[i], cmap='binary')
        plt.sca(ax)
        plt.title(fra + i + 1)
    plt.show()
    
def printOnePicture(i):
    image[i] = Y0[:,i].reshape(28, 28)
    fig = plt.figure()
    plt.imshow(image[i], cmap='binary')
    plt.title(i)
