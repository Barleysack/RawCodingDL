import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [['barber', 'person'], ['barber', 'good', 'person'], 
['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'],
 ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'],
  ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
   ['barber', 'went', 'huge', 'mountain']]
#정수 인코딩 ON
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.
encoded = tokenizer.texts_to_sequences(sentences)
max_len = max(len(item) for item in encoded) #동일한 길이로 맞추기 위해서, 이중 가장 길이가 긴 문장의 길이를 계산. 

for item in encoded: # 각 문장에 대해서
    while len(item) < max_len:   # max_len보다 작으면
        item.append(0)

padded_np = np.array(encoded)
print(padded_np)

"""길이가 7보다 짧은 문장에는 전부 숫자 0이 뒤로 붙어서 모든 문장의 길이가 전부 7이된 것을 알 수 있습니다. 기계는 이제 이들을 하나의 행렬로 보고, 
병렬 처리를 할 수 있습니다. 또한, 0번 단어는 사실 아무런 의미도 없는 단어이기 때문에 자연어 처리하는 과정에서 기계는 0번 단어를 무시하게 될 것입니다. 
이와 같이 데이터에 특정 값을 채워서 데이터의 크기(shape)를 조정하는 것을 패딩(padding)이라고 합니다. 
숫자 0을 사용하고 있다면 제로 패딩(zero padding)이라고 합니다."""

#내가 CV에서 배웠던 제로패딩과는 느낌이 사뭇 다르지요? ㅋㅋ


from tensorflow.keras.preprocessing.sequence import pad_sequences 
#케라스쪽 api 달달.
encoded = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(encoded) #기본 값으로 앞쪽에 0을 채우는 것을 방법으로 한다
#뒤에 0을 채우고 싶다면 인자로 padding = 'post'를.

#정해진 길이로 하고 싶다면?
padded = pad_sequences(encoded, padding = 'post', maxlen = 5)

