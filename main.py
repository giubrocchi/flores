import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, GlobalMaxPooling2D, BatchNormalization, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
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
  image = cv2.resize(image, (200, 200))/255
  images.append(image)
  labels.append(current_label)

  count += 1
  if(count % 80 == 0):
    current_label += 1

train, test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

y_train = to_categorical(y_train, 17)
y_test = to_categorical(y_test, 17)

x_train = tf.data.Dataset.from_tensor_slices((train, y_train))
x_train = x_train.batch(16)

x_test = tf.data.Dataset.from_tensor_slices((test, y_test))
x_test = x_test.batch(16)

augmentation = Sequential([RandomFlip("horizontal"), RandomRotation(0.25), RandomZoom(0.2, 0.2)])

trained = VGG16(input_shape = (200, 200, 3), include_top = False, weights = 'imagenet')
trained.trainable = False

conv = Conv2D(32, (3, 3), padding='same', activation='relu')
pool = GlobalMaxPooling2D()
norm = BatchNormalization()
drop = Dropout(0.3)
outputs = Dense(17, activation="softmax", dtype='float32')

model = Sequential([augmentation, trained, conv, pool, norm, drop, outputs])
model.compile(Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics='accuracy')

history = model.fit(x_train, epochs=30, validation_data=x_test, verbose=1)

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
