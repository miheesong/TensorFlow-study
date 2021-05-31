import numpy as np
from tensorflow.python.eager.monitoring import Metric

#1. 데이터
x_train = np.array(range(1,11))  # train
y_train = np.array(range(1, 11))
x_test = np.array([11, 12, 13, 14, 15])  # evaluate
y_test = np.array([11, 12, 13, 14, 15])
x_pred = np.array([16, 17, 18]) # x_test로 y_pred 만든다.


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(5, input_dim=2))
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측 [[11,12,13],[21,22,23]]
results = model.evaluate(x_test, y_test)
print('results : ', results)

# y_predict = [[11,12,13],[21,22,23]]
# y_predict = np.transpose(y_predict)
# y_new_predict = model.predict(y_predict)
# print("y_predict : ", y_new_predict)
y_predict = model.predict(x_test) # 평가하는 데이터로 예측해야함.


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 예측값과 결과값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score

R2 = r2_score(y_test, y_predict)
print('R2 : ', R2)