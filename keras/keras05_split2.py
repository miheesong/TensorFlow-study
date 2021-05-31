from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

#1. 데이터
x = np.array(range(1, 101))
y = np.array(range(101, 201))

# x_train = x[:60], x_val = x[60:80], x_test = x[80:]
# y_train = y[:60], y_val = y[60:80], y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.4, train_size=0.6  #, shuffle=True  # shuffle default : true
)
x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, train_size=0.5  #, shuffle=True  # shuffle default : true
)
# print(x_train)
# print(x_test)
# print(x_val)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([101,102,103])
print("y_predict : ", y_predict)