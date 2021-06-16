# 과제 : embedding 빼고 모델 완성 (13,5) -> reshape(13,5,1)

# embedding과정-> (100만)단어가 있다면 데이터가 너무 많아짐 
# 가까운 벡터끼리 수치화 (100만, 100만) -> (100만,2) 좌표로 나타낸다
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재미있어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요', '글세요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다.', '참 재밋네요', '현욱이가 잘 생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x) # shape을 제일 큰걸로 맞춰준다. [2,4] -> [0,0,0,2,4] 시계열 데이터는 마지막에 더 비중을준다.

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5) # pad_x : train data # pre:앞부터0채움 post:뒤부터

print(pad_x)
print(pad_x.shape) # (13,5)

print(np.unique(pad_x))
print(len(np.unique(pad_x))) # 28

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim=28, output_dim=7, input_length=5)) # 수치화됨. input_dim : 사전의 개수
# model.add(Embedding(28,7)) # input_length 없으면 알아서 찾아준다.
# model.add(LSTM(32))
# model.add(Conv1D(32, 3))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 7)              196
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5120
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,349
# Trainable params: 5,349
# Non-trainable params: 0
# _________________________________________________________________

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, None, 7)           196
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                5120
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 5,349
# Trainable params: 5,349
# Non-trainable params: 0
# _________________________________________________________________

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 7)              196
# _________________________________________________________________
# conv1d (Conv1D)              (None, 3, 32)             704
# _________________________________________________________________
# dense (Dense)                (None, 3, 1)              33
# =================================================================
# Total params: 933
# Trainable params: 933
# Non-trainable params: 0
# _________________________________________________________________

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(pad_x, labels,epochs=100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)