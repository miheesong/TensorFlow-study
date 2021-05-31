# 다:다 mlp
import numpy as np

#1. 데이터
x = np.array([range(100), range(301, 401), range(1,101)]) # (3, 100)
y = np.array([range(711, 811), range(1, 101), range(201, 301)]) # (100, )

x = np.transpose(x) # (100, 3)
y = np.transpose(y) # (100, 3)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(5, input_dim=2))
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(3))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

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
