import numpy as np

a = np.array(range(1,11))
size = 6

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size) # 0-4:x 5:y
print(dataset) 

x = dataset[:, :size-1] # 앞에는 행, 뒤는 열
y = dataset[:, size-1]
print(x)
print(y)

x_pred = [[6,7,8,9,10]]

print(x.shape)
print(y.shape)
x = x.reshape(6,4,1)
#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, input_shape=(5, 1)))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)

#4. 평가 예측
results = model.evaluate(x, y)
print('loss : ', results)

x_pred = x_pred.reshape(1,5)

y_pred = model.predict(x_pred)
print("result : ", y_pred)
