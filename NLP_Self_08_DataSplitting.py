import pandas as pd


#아시다시피, 데이터셋을 학습 데이터와 벨리데이션, 테스트 데이터로 나누셔야합니다.
#이를 위해 리스트 형태의 자료를 x와 y로 zip을 활용해 나눠보겠다.

x,y = zip(['a',1],['b',2],['c',3])
print(x)
print(y)
sequences=[['a', 1], ['b', 2], ['c', 3]] # 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
X,y = zip(*sequences) # *를 추가
print(X)
print(y)


values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
X=df['메일 본문']
y=df['스팸 메일 유무']
print(X)
print(y)
#넘파이를 이용한 분리.
import numpy as np

ar = np.arange(0,16).reshape((4,4))
print(ar)

X=ar[:, :3]
print(X)
y=ar[:,3]
print(y)

#간만에 보는 사이킷런.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1234)\

"""
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
print(X)
print(list(y)) #레이블 데이터
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
#3분의 1만 test 데이터로 지정.
#random_state 지정으로 인해 순서가 섞인 채로 훈련 데이터와 테스트 데이터가 나눠진다.