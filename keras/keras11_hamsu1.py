# 함수형 모델

import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1,101),
              range(100), range(301, 401)]) # (5, 100)
y = np.array([range(711, 811), range(1, 101)]) # (2, 100)

x = np.transpose(x) # (100, 5)
y = np.transpose(y) # (100, 2)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input #함수형에서 쓰일 input layer

# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary() # param : bias 더한값 출력

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
output1 = Dense(2)(dense2)
model = Model(inputs=input1, outputs=output1)
model.summary()

# Sequential, 함수형 모델은 같다.


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1,verbose=2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = model.predict(x_test) # 평가하는 데이터로 예측해야함.


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 예측값과 결과값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score

R2 = r2_score(y_test, y_predict)
print('R2 : ', R2)
