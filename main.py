import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Input, Model
from keras.optimizers import Adam
from keras.layers import Normalization, Conv2D, GlobalMaxPooling2D, BatchNormalization, Dense
import cv2
import os

'''
[0] = Daffodil      [1] = Snowdrop      [2] = Lily Valley     [3] = Bluebell
[4] = Crocus        [5] = Iris          [6] = Tigerlily       [7] = Tulip
[8] = Fritillary    [9] = Sunflower     [10] = Daisy          [11] = Colts' Foot
[12] = Dandelion    [13] = Cowslip      [14] = Buttercup      [15] = Windflower
[16] = Pansy
'''

labels = []
current_label = 0
count = 0
images = []

for path in os.listdir('./jpg'):
  image = cv2.imread('./jpg/' + path)[...,::-1]
  image = cv2.resize(image, (500,500))
  images.append(image)
  labels.append(current_label)

  count += 1
  if(count % 80 == 0):
    current_label += 1

train, test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

x_train = tf.data.Dataset.from_tensor_slices((train, y_train))
x_train = x_train.shuffle(len(x_train)).batch(16)
x_test = tf.data.Dataset.from_tensor_slices((test, y_test))
x_test = x_test.shuffle(len(x_test)).batch(16)

input = Input(shape=(500, 500, 3))
x = Normalization()(input)
x = Conv2D(90, (10, 10), padding='same', activation='relu')(x)
x = GlobalMaxPooling2D()(x)
x = BatchNormalization()(x)
outputs = Dense(17, activation="softmax", dtype='float32')(x)

model = Model(input, outputs)
model.compile(Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics='accuracy')

class Callbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('accuracy') >= 0.85:
      self.model.stop_training = True

callback = Callbacks()

history = model.fit(x_train, epochs=80, validation_data=x_test, verbose=1, callbacks=[callback])

model.save('./model')

plt.figure(1)
plt.plot(history.history['accuracy'][1:],label='Train')
plt.plot(history.history['val_accuracy'][1:],label='Valid')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure(2)
plt.plot(history.history['loss'][1:],label='Train')
plt.plot(history.history['val_loss'][1:],label='Valid')
plt.xlabel('Epoch')
plt.title('Loss')
plt.legend()
plt.show()