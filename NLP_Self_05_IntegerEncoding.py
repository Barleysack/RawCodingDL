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

print(vocab) #중복을 제거해두고 사용 빈도에 따라 정리해둔것이다. 단어가 키로, 빈도가 값으로 저장되어 있다.
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True) #빈도에 따른 정렬

#이제 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여합니다.
word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word] = i
print(word_to_index)


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
print(word_to_index)

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
print(encoded)