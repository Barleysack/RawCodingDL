import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords

text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화.
text = sent_tokenize(text)

# 정제와 단어 토큰화
vocab = {} # 파이썬의 dictionary 자료형
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    sentences.append(result) 

"""텍스트를 숫자로 바꾸는 단계라는 것은 본격적으로 자연어 처리 작업에 들어간다는 의미이므로, 단어가 텍스트일 때만
할 수 있는 최대한의 전처리를 끝내놓아야 합니다. 위의 코드를 보면, 동일한 단어가 대문자로 표기되었다는 이유로 서로
다른 단어로 카운트되는 일이 없도록 모든 단어를 소문자로 바꾸었습니다. 
그리고 자연어 처리에서 크게 의미를 갖지 못하는 불용어와 길이가 짧은 단어를 제거하는 방법을 사용하였습니다."""

# print(vocab) #중복을 제거해두고 사용 빈도에 따라 정리해둔것이다. 단어가 키로, 빈도가 값으로 저장되어 있다.
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) #빈도에 따른 정렬

#이제 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여합니다.
word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word] = i
# print(word_to_index)


"""1의 인덱스를 가진 단어가 가장 빈도수가 높은 단어가 됩니다. 
그리고 이러한 작업을 수행하는 동시에 각 단어의 빈도수를 알 경우에만 할 수 있는 전처리인
 빈도수가 적은 단어를 제외시키는 작업을 합니다. 등장 빈도가 낮은 단어는 자연어 처리에서 의미를 가지지 않을 
 가능성이 높기 때문입니다. 여기서는 빈도수가 1인 단어들은 전부 제외시켰습니다.

자연어 처리를 하다보면, 텍스트 데이터에 있는 단어를 모두 사용하기 보다는 빈도수가 
가장 높은 n개의 단어만 사용하고 싶은 경우가 많습니다. 위 단어들은 빈도수가 높은 순으로 낮은 정수가
부여되어져 있으므로 빈도수 상위 n개의 단어만 사용하고 싶다고하면 vocab에서 정수값이 1부터 n까지인 단어들만 사용하면 됩니다
여기서는 상위 5개 단어만 사용한다고 가정하겠습니다."""

vocab_size = 5
words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
# print(word_to_index)

#단어 집합에 없는 단어들을 Out-Of-Vocab의 약자로 OOV라 하며,
# word_to_index에 추가해 해당 키 아래에 달아둔다.

word_to_index['OOV'] = len(word_to_index) + 1
#sentences의 모든 단어를 매핑된 정수로 인코딩한다.

encoded = []
for s in sentences:
    temp = []
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
# print(encoded)

#Counter의 사용. 위에보다 훨씬 쉽죠?


from collections import Counter


words = sum(sentences,[]) #하나의 리스트로 합친다. 이렇게도 합쳐지는구나. 리스트 자체가 원소이니 당연한거지만.
vocab = Counter(words) #단어 빈도수로 정리 완
vocab_size = 5
vocab = vocab.most_common(vocab_size)

word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i+1
    word_to_index[word] = i
# print(word_to_index) 
#높은 빈도수를 가진 단어일 수록 낮은 정수 인덱스를 부여

##NLTK의 빈도수 계산 도구인 FreqDist(). 
#거의 카운터처럼 써먹으면 된디야~

from nltk import FreqDist
import numpy as np

#np.hstack으로 문장 구분 제거 후 입력으로 사용. 

#hstack 이거 뭔가 했더니 라인 86, sum(sentences)랑 동일한 기능일세 그래
vocab = FreqDist(np.hstack(sentences))
print(vocab)

print(vocab["barber"]) #barber는 몇번? 이 되겠지

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
print(vocab) #동일한 처리... 저 most_common은 어디거였지? 카운터 거일려나?


word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
#오, 파이써닉 코드가 나왔다. enum을 사용했는데? 이를 통해서 인덱스를 부여했습니다요
#enum은 순서가 있는 자료형을 받아서 입력 순서대로 인덱스를 순차적으로 반환한다는 특성이 존재. 


print(word_to_index)


#이번엔 케라스로 해보자!
from tensorflow.keras.preprocessing.text import Tokenizer

sentences=[['barber', 'person'], 
['barber', 'good', 'person'], 
['barber', 'huge', 'person'], 
['knew', 'secret'], 
['secret', 'kept', 'huge', 'secret'], 
['huge', 'secret'], ['barber', 'kept', 'word'], 
['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
 ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], 
 ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.
print(tokenizer.word_index) #이렇게만 해도 빈도수대로 인덱스가 먹히는구만,,,? 케라스 당신은 대체..
#각 단어가 카운트를 수행하였을 때 몇 개였는지를 보고자 한다면 word_counts를 사용합니다.
print(tokenizer.word_counts)
#나는 개인적으로 케라스가 더 직관적인거같음.
#근데 왜 파이토치가 더 큰다고 했던걸까?
print(tokenizer.texts_to_sequences(sentences))
#해당 단어를 해당 인덱스로 먹인다. 
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(sentences)
#상위 단어 사용을 위한 작은 재정의 .
#패딩에 관한 내용은 이후 단원에서 읽어봅시다. 
#이렇게 말을 해봐야, WORD_INDEX나 WORD_COUNT에서는 뭐 별말 없음.
print(tokenizer.texts_to_sequences(sentences)) #여기서 실제 적용. 

#케라스 토크나이저는 WORD TO INDEX 과정에서 OOV는 날려먹는다.
#보존하고 싶다면 oov_token 인자를 참고합시다 .
#oov의 인덱스는 1로 설정이 된다. 

