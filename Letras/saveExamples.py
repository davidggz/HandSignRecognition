import pandas as pd
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
import cv2
from PIL import Image

path_npy = "DatasetsIAO/Letras/"
letras_train_X = np.asarray(np.load(path_npy + "letras_train_X.npy"))

letras_test_X = np.asarray(np.load(path_npy + "letras_test_X.npy"))

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
for example in [0, 1500, 3000, 4500, 6000]:
    imagenesNumeros = letras_train_X.reshape(-1, 28, 28, 1)
    imagen = letras_train_X[example]
    imagen = imagen.reshape(28, 28)

    plt.imshow(imagen, cmap="gray")
    plt.savefig("images/ejemploTrainGRAF_"+str(example)+".png")

    svimg=Image.fromarray(imagen.astype('uint8'))
    svimg.save("images/ejemploTrain_"+str(example)+".png")

from keras.preprocessing import image
for example in range(0, 10):
    imagenesNumeros = letras_test_X.reshape(-1, 28, 28, 1)
    imagen = letras_train_X[example]
    imagen = imagen.reshape(28, 28)

    plt.imshow(imagen, cmap="gray")
    plt.savefig("images/ejemploTestGRAF_"+str(example)+".png")

    svimg=Image.fromarray(imagen.astype('uint8'))
    svimg.save("images/ejemploTest_"+str(example)+".png")

