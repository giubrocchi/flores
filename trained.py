import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import numpy as np

labels = [
  'Daffodil', 'Snowdrop', 'Lily Valley', 'Bluebell', 'Crocus', 'Iris',
  'Tigerlily', 'Tulip', 'Fritillary', 'Sunflower', 'Daisy', 'Coltsfoot', 'Dandelion',
  'Cowslip', 'Buttercup', 'Windflower', 'Pansy'
]
model = tf.keras.models.load_model('./model')

for path in os.listdir('./testes'):
  image = cv2.imread('./testes/' + path)[...,::-1]
  image = cv2.resize(image, (200,200))
  image = np.array(image)
  image = np.expand_dims(image, axis=0)
  predict = model.predict(image)

  figure = plt.figure(figsize=(2, 2))
  image = np.squeeze(image, axis=0)
  plt.imshow(image)
  plt.subplots_adjust(top=.7)
  plt.axis('off')
  plt.title(f'Real name: {path}\nPrediction: {labels[np.argmax(predict[0], axis=0)]}')
  plt.show()
