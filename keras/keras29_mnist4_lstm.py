import numpy as np

#1. Data
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.

# x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
# x_test = x_test.reshape(10000,28,28,1).astype('float32')/255
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. Model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM

model = Sequential()
# model.add(Dense(256, activation='relu', input_shape=(784,)))
# model.add(Dense(128))
model.add(LSTM(10, activation='relu', input_shape=(28,28))) # input_shape=(784,1)
model.add(Dense(50,activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. Compile, Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, epochs=2, verbose=2, validation_split=0.2)

#4. Evaluate, Predict
y_pred = model.predict(x_test)
loss = model.evaluate(x_test,y_test)
print('loss :', loss[0])
print('acc  :', loss[1])

# loss: 0.1612 - acc: 0.9788
# loss : 0.16121423244476318
# acc  : 0.9787999987602234