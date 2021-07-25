import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt

from pandas_profiling.profile_report import ProfileReport

loc = pd.Series([1000,20000,3030,9295],index=["a","b","c","d"]) #1차원 배열값 with index, 이것이 Series.


values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)


data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]
df = pd.DataFrame(data)


#데이터프레임의 형성. 
#이렇게 맹글어도 되고 저렇게 맹글어도 되고~ 알아두고 말자 
#프리코스 04에서 했지만, 복습하는 김에 다시 보자. 
#df.head(n) 앞 부분을 n개만 선택.
#df.tail(n) 뒷 부분을 n개만 선택.
#df['컬럼네임'] 해당하는 열 보기

a = np.array([1, 2, 3, 4, 5]) #리스트 받아서 넘파이 오브젝트로 생-성 
b = np.array([[10, 20, 30], [ 60, 70, 80]]) #

#ndarray를 만드는 여러가지 방법을 봅시다 . 
a = np.zeros((2,3))
a = np.ones((2,3))
a = np.eye((3))
a = np.full((2,3),20)
a = np.random.random((2,2))
#np.arange. 알지?
a = np.array(np.arange(30)).reshape((5,6))



#numpy 슬라이싱 : 리스트랑 비스무리~합니다요 
a = np.array([[1,2], [4,5], [7,8]])
b = a[[2, 1],[1, 0]] # a[[row2, row1],[col1, col0]]을 의미함.

x = np.array([1,2,3])
y = np.array([4,5,6])

b = x + y # 각 요소에 대해서 더함
# b = np.add(x, y)와 동일함

b = x - y # 각 요소에 대해서 빼기
# b = np.subtract(x, y)와 동일함

b = b * x # 각 요소에 대해서 곱셈
# b = np.multiply(b, x)와 동일함

b = b / x # 각 요소에 대해서 나눗셈
# b = np.divide(b, x)와 동일함


a = np.array([[2,1],[1,2]])
b = np.array([[2,1],[2,2]])

c = np.dot(a, b) #행렬곱을 위한 닷프로세스.
print(c)
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
 

#축레이블 삽입 
plt.title('test')
plt.plot([1,2,3,4],[2,4,8,6])
plt.xlabel('hours')



"""좋은 머신러닝 결과를 위하여 데이터의 성격을 파악하는 과정 .
데이터 내 값의 분포, 변수간의 관계 , Null 값과 같은 결측값 존재 유무 파악
EDA(Exploratory Data Analysis, 탐색적 데이터 분석)
"""
data = pd.read_csv('./Datasets/spam.csv',encoding='latin1')
rpt = ProfileReport(data,title = "pandas")
#근데 어째 이게 잘 안되는구만...? #파이토치 프로파일링을 이용하는 편이 나을듯.