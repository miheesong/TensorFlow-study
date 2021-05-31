import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[10, 85, 70], [90, 85, 100], 
             [80, 50, 30], [43, 60, 100]])        # (4, 3)

y = np.array([75, 65, 33, 85])  # (4, )

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(3,)))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # metrics 안에 지표를 넣는다. 
# metrics는 훈련에 들어가는게 아니라 mae를 넣으면 이런값이 나온다는 알려줌
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x, y) # loss에 metrics값도 반환한다.
print('loss : ', loss)

