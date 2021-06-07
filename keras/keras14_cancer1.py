# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=50, validation_split=0.2)

results = model.evaluate(x, y)
print('loss : ', results[0])
print('metrics : ', results[1])

y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1])

# [[ 6.602269 ]
#  [-2.528346 ]
#  [-2.316661 ]
#  [-3.770442 ]]
# [0 0 0 0]



# train split 분류하고 layer늘려서 acc 나오게 튜닝.