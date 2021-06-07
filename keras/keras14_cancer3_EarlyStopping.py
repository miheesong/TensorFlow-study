# 이진분류
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)


x = datasets.data
y = datasets.target

# from tensorflow.keras.utils import to_categorical

# y = to_categorical(y)
# print(x.shape, y.shape) # (569, 30) (569,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping (monitor='val_loss', patience=5, mode='min') # min,max인지 모를땐 -> auto ,mode:max -> acc일때 사용.

model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=2, 
          callbacks=[early_stopping])

results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('metrics : ', results[1])

y_pred = model.predict(x_test[-5:-1])
print(y_pred)
print(y_test[-5:-1])

# [[ 6.602269 ]
#  [-2.528346 ]
#  [-2.316661 ]
#  [-3.770442 ]]
# [0 0 0 0]
