import matplotlib.pyplot as plt
import keras

model = keras.models.load_model("../Letras/0024-HSRL-CMCMCMFDrDD-0009.h5")
model.summary()
from keras.preprocessing import image

import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import os, shutil
from keras import backend as K

j = 1
for fname in [0,1500, 3000, 4500, 6000]:
  IMG_SIZE = 28
  path = "../images/ejemplo" + str(fname) + ".png"
  img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
  x = image.img_to_array(img)
  x = x.reshape(1, IMG_SIZE, IMG_SIZE, 1)
  x = x.astype('float32')
  x /= 255.0

  preds = model.predict(x)
  category = np.argmax(preds)
  print(category)
  output = model.output[:, category]
  #Ultima convolucional del modelo
  last_conv_layer = model.get_layer('conv2d_3')

  grads = K.gradients(output, last_conv_layer.output)[0]
  print(grads)
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  print(pooled_grads)
  iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
  pooled_grads_value, conv_layer_output_value = iterate([x])

  for i in range(64):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
  print(conv_layer_output_value)
  heatmap = np.mean(conv_layer_output_value, axis=-1)
  print(heatmap)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  print(heatmap)
  cv2.imwrite("heatmapExtra"+str(j)+".jpg", heatmap)
  plt.imshow(heatmap)
  plt.savefig("heatmap"+ str(j) + ".jpg")

  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  img = img.reshape(28, 28, 1)
  print(img.shape)
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

  plt.imshow(heatmap)
  plt.savefig("heatmapMODIF" + str(j) + ".jpg")
  superimposed_img = heatmap * 0.4 + img

  cv2.imwrite("EXTRA_"+str(j)+".jpg", superimposed_img)
  plt.imshow((superimposed_img * 255).astype(np.uint8))
  plt.savefig("imposed" + str(j) + ".jpg")
  j = j + 1
