
# !pip3 install python-mnist


import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
import MNIST
import sys

argumentList = sys.argv


mndata = MNIST(argumentList[1])

train_images, train_labels = mndata.load_training()

test_images, test_labels = mndata.load_testing()

train_images=np.array(train_images)
test_images=np.array(test_images)

print(train_images.shape,test_images.shape)

# !pip install tensorflow==1.14.0
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D



x_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
x_test = test_images.reshape(test_images.shape[0], 28, 28, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255
x_test /= 255
input_shape = (28, 28, 1)


model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten()) 
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=np.array(train_labels) , epochs=15)



p=model.predict_classes(x_test)
for i in p:
  print(i)
