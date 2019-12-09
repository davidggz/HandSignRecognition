import pandas as pd
import numpy as np
import keras as keras
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

NAME = "0016-HSRL-CMCMCMFDD-0001"

letras_train_X = np.load("../DatasetsIAO/Letras/letras_trainV1_X.npy")
letras_train_Y = np.load("../DatasetsIAO/Letras/letras_trainV1_Y.npy")
letras_test_X = np.load("../DatasetsIAO/Letras/letras_testV1_X.npy")
letras_test_Y = np.load("../DatasetsIAO/Letras/letras_testV1_Y.npy")
letras_val_X = np.load("../DatasetsIAO/Letras/letras_valV1_X.npy")
letras_val_Y = np.load("../DatasetsIAO/Letras/letras_valV1_Y.npy")

# Modelo de red neuronal
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.25))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(25, activation='softmax'))
          
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(letras_train_X, letras_train_Y, batch_size=32)
val_generator = val_datagen.flow(letras_val_X, letras_val_Y, batch_size=32)

from keras import optimizers

model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4), 
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=letras_train_X.shape[0]//32,
    epochs=20,
    validation_data=val_generator,
    validation_steps=letras_val_X.shape[0]//32
    )

model.save(NAME + ".h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.clf()
plt.grid(linestyle='-', linewidth=1.6, alpha=0.3)
plt.plot(epochs, acc, 'b', linewidth=1.0, label='Training acc')
plt.plot(epochs, val_acc, 'r', linewidth=1.0, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(NAME + "-acc.png")

plt.clf()
plt.grid(linestyle='-', linewidth=1.6, alpha=0.3)
plt.plot(epochs, loss, 'b', linewidth=1.0, label='Training loss')
plt.plot(epochs, val_loss, 'r', linewidth=1.0, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(NAME + "-loss.png")

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
from sklearn.metrics import  confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

letras_test_X = letras_test_X.astype('float32')
letras_test_X /= 255.0

labels = np.where(letras_test_Y==1)[1]
pred = model.predict(letras_test_X, verbose=1)

pred = np.argmax(pred, axis=1)

#Ahora puedo hacer la matriz de confusion
axisName = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
cm = confusion_matrix(labels, pred)
cm_sn = pd.DataFrame(cm, columns=axisName, index=axisName)

plt.figure(figsize=(20,10))
sns.heatmap(cm_sn, annot=True, cmap='Blues', fmt='g')
plt.title(NAME + '\nAccuracy:{0:.3f}'.format(accuracy_score(labels, pred)))
plt.ylabel('True value')
plt.xlabel('Predicted value')
plt.savefig(NAME + "-cm.png")  

print("Test acc: " + str(accuracy_score(labels,pred)))                                      
