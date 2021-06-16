from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
# {'진짜': 1, '마구': 2, '나는': 3, '맛있는': 4, '밥을': 5, '먹었다': 6}
# 많이 나온단어, 앞에 나온단어 순으로

x = token.texts_to_sequences([text])    #어절순으로 잘라서 수치화
# [[3, 1, 1, 4, 5, 2, 2, 6]]
# print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)   # 6
print(word_size)

x = to_categorical(x)

print(x)
print(x.shape) #(1,8,7) 0부터 시작, 위에는 1부터 시작해서 0이 자동으로 채워짐.