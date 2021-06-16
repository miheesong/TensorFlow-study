import numpy as np

a = np.array(range(1,11))
size = 5

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

#2. 모델
#3. 컴파일, 훈련
#4. 평가 예측
# 
#  Dense로 만들기!!!