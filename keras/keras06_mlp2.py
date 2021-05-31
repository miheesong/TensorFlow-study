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


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측 
loss = model.evaluate(x, y)
print('loss : ', loss)

