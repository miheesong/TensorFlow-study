from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), strides=1, input_shape=(5,5,1)))
#                        이미지 자른 크기(가로,세로), strides : 1칸씩 이동
model.add(Conv2D(5, (2,2), padding='same'))

model.add(Flatten()) # dense에 넣기위해 Flatten, 수치를 펼친다. 
# 이미지가 어떤건지 맞추기위해 최종 레이어에 통과시키기 위해 flatten통과
model.add(Dense(1)) # 수치계산땜에 dense 1

model.summary()

'''
param이 왜 50이고 205개이고 total params 336인지

커널 사이즈 ( 2*2), filter (10), 10 filter에 대한  bias

(2*2) * 10 + 10 = 50

10(이전 channel) * (2*2) * 5(filter) + 5 = 205

flatten 80 개에 대한 dense1의 param = 80 + 1(bias) = 81

----> 50 + 205 + 81 = 336
'''

