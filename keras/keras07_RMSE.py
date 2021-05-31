import numpy as np
from tensorflow.python.eager.monitoring import Metric

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
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측 [[11,12,13],[21,22,23]]
results = model.evaluate(x, y)
print('results : ', results)

# y_predict = [[11,12,13],[21,22,23]]
# y_predict = np.transpose(y_predict)
# y_new_predict = model.predict(y_predict)
# print("y_predict : ", y_new_predict)
y_predict = model.predict(x) # x 전체에 대한 예측값 10개


from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict): # 예측값과 결과값
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y, y_predict))
print("mse : ", mean_squared_error(y, y_predict))