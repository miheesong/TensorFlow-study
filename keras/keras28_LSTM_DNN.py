import numpy as np

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)
print(y.shape)

# LSTM을 DENSE로 바꾸기위해 모양만 바꿔준다.
# x = x.reshape(13,3,1)
# print(x.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))
model.add(Dense(64, input_shape=(3,), activation='relu')) # 데이터가 적어서 더 좋게 느낄수있다.
model.add(Dense(32, activation='relu')) # 노드수를 2진법으로 넣는게 좋다.
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가, 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_pred = np.array([50,60,70]) # (3, )  -> (1,3,1) [[[5], [6], [7]]]
x_pred = x_pred.reshape(1,3)

y_pred = model.predict(x_pred)
print("result : ", y_pred) # 80.08237
