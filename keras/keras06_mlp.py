import numpy as np

#1. 데이터
x = np.array([range(1,11), range(11,21)])
y = np.array(range(1, 11))

#print(x.shape)
x = np.transpose(x)
# print(x.shape)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(5, input_dim=2))
model.add(Dense(10, input_shape=(2,)))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일,훈련

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2, train_size=0.8  #, shuffle=True  # shuffle default : true
)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
          validation_split= 0.25)

#4. 평가, 예측 [[11,12,13],[21,22,23]]
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = [[11,12,13],[21,22,23]]
y_predict = np.transpose(y_predict)
y_new_predict = model.predict(y_predict)
print("y_predict : ", y_new_predict)