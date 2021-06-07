import numpy as np

from sklearn.datasets import load_boston 

data = load_boston()
x = data.data
y = data.target # label 으로도 쓰기도함

# print(x) # (506, 13)
# print(y) # (506, 1) y 는 집값을 담은 스칼라 1차원 벡터
# print(data.feature_names)
# print(data.DESCR) # Description (데이터셋에 대한 자세한 설명 나옴)


# column = feature = attribute .....
# row = instance ....

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1) # hidden layer 의 첫번째 layer
dense2 = Dense(40)(dense1)
dense3 = Dense(30)(dense2) 
output1 = Dense(1)(dense3) 

model = Model(inputs=input1, outputs=output1)

# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=300, verbose=5)

# 예상 및 결과
result = model.evaluate(x_test, y_test, batch_size=1)
print("result : ", result)

y_predict = model.predict(x_test) 

from sklearn.metrics import mean_squared_error # mse 를 의미함

def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score

print('R2 : ', r2_score(y_test, y_predict))