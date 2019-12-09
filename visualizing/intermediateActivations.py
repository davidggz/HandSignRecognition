from keras.models import load_model
model = load_model('../Letras/0024-HSRL-CMCMCMFDrDD-0009.h5')
model.summary()
img_path = '../images/ejemplo0.png'
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt
img = img_tensor.reshape(28, 28)
plt.imshow(img, cmap = "gray")
plt.savefig('selectedImg.jpg')

from keras import models
layer_outputs = [layer.output for layer in model.layers[:6]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)
import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.savefig("primeraCapaViridis5.jpg")
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.savefig("primeracapaViridis10.jpg")
#plt.imsave('primeraCapaViridis5.jpg',first_layer_activation[0, :, :, 5], cmap='viridis')
#plt.imsave('primeraCapaViridis10.jpg',first_layer_activation[0, :, :, 10], cmap='viridis')

layer_names = []
for layer in model.layers[:6]:
  layer_names.append(layer.name)
print(layer_names)
images_per_row = 8

for layer_name, layer_activation in zip(layer_names, activations):
  n_features = layer_activation.shape[-1]
  size = layer_activation.shape[1]
  n_cols = n_features // images_per_row
  display_grid = np.zeros((size * n_cols, images_per_row * size))

  for col in range(n_cols):
    for row in range(images_per_row):
      channel_image = layer_activation[0,:, :,col * images_per_row + row]
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')
      display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

  scale = 1. / size
  plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid, aspect='auto', cmap='viridis')
  plt.savefig('display_grid_' + layer_name + '_.jpg')
