import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
        #   validation_data=(x_validation, y_validation))
          validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([9])
print("result : ", result)