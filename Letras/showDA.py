import pandas as pd
import numpy as np
import keras as keras
import matplotlib.pyplot as plt
path_npy = "../DatasetsIAO/Letras/"
letras_train_X = np.load(path_npy + "letras_trainV2_X.npy")

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1)

fig = plt.figure(figsize = (8, 8))
pos = 1

from keras.preprocessing import image
for example in [0, 1500, 3000, 4500, 6000]:
    i = 0
    x = letras_train_X[example]
    x = x.reshape(1, 28, 28, 1)
    ax = fig.add_subplot(5, 5, pos)
    ax = plt.imshow(x.reshape(28, 28), cmap='gray', interpolation='nearest')
    plt.axis('off')
    pos += 1

    for batch in train_datagen.flow(x, batch_size=1):
       ax = fig.add_subplot(5, 5, pos)
       ax = plt.imshow(image.array_to_img(batch[0]), cmap='gray', interpolation='nearest')
       plt.axis('off')
       # plt.savefig("EjemploDA" + str(example) + "_" + str(i) + ".png")
       i += 1
       pos += 1
       if i % 4 == 0:
           break

plt.savefig("images/imagenesAumentadas.png", bbox_inches='tight')
