import numpy as np
import tensorflow as tf

#1. 정제된 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  ##dense : y=wx+b를 나타냄

model = Sequential()
model.add(Dense(30, input_dim=1))  #1개의 노드를 input, 다음 레이어엔 3개가 나간다.
model.add(Dense(40)) #1개의 레이어 추가, 4:노드의 개수
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss를 최적화 시키는것
model.fit(x, y, epochs=40, batch_size=1) #몰아서 하면 빠르다. epochs 훈련 10번
# 배치를 작게하면 성능이 좋다. 배치를 얼마로 잡아야하나는 개발자에게 달려있음.

#4. 평가, 예측
loss = model.evaluate(x,y) # 학습에 사용한 데이터를 평가에 쓸수없다.
print('loss : ', loss)

results = model.predict([4])
print('results : ', results)

## result가 잘안나오면 batch_size를 넣고, epochs 늘리고, node수 증가시킨다.
