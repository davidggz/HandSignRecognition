import matplotlib.pyplot as plt
from keras import backend as K

import keras

model = keras.models.load_model("")
model.summary()
from keras.preprocessing import image

import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os, shutil

j = 1
for fname in [0,1500, 3000, 4500, 6000]:
  IMG_SIZE = 28
  img = image.load_img(fname, target_size=(IMG_SIZE, IMG_SIZE))
  x = image.img_to_array(img)
  x = x.reshape(1, IMG_SIZE, IMG_SIZE, 1)
  x = x.astype('float32')
  x /= 255.0

  preds = model.predict(x)
  print('Predicted:', preds)

  print(np.argmax(preds))

  output = model.output[:]
  #Ultima convolucional del modelo
  last_conv_layer = model.get_layer('conv2d_4')

  grads = K.gradients(output, last_conv_layer.output)[0]
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  iterate = K.function([model.input, K.learning_phase()],[pooled_grads, last_conv_layer.output[0]])
  pooled_grads_value, conv_layer_output_value = iterate([x, 0])

  for i in range(128):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  heatmap = np.mean(conv_layer_output_value, axis=-1)

  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)

  cv2.imwrite("heatmapExtra"+str(j)+".jpg", heatmap)

  img = cv2.imread(fname)
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = heatmap * 0.4 + img

  cv2.imwrite("EXTRA_"+str(j)+".jpg", superimposed_img)
  j = j + 1
