#matplotlib의 기본적인 내용
#pyplot 객체를 사용해 데이터를 표시
#pyplot 객체에 그래프들을 쌓은 다음 show로 flush

import matplotlib.pyplot as plt
import numpy as np

#단점?: argument를 kwargs로 받아서; 고정된 아규먼트가 없기에 알트탭으로 확인이 어려움. 
#따로 찾아보면 오버로딩된게 한 네개즘 있는듯


x= range(100)
y= [value**2 for value in x]
plt.plot(x,y)

#모판의 이름: figure.
a = range(100)
b = range(100)
plt.plot(a,b)
plt.show()
#show의 순간 flush 되는것이다. 메모리 플러쉬가 일어난다고 생각하자. 

#Color: RGB의 값이 들어감. color도 되고 그냥 c로도 가능. 그냥 검색해서 컬러 잡아넣어도 적당하다 ㅋㅋ 
#linestyle같은건 뭐 나중에 찾아보슈 
#set legend: 범례 설정.
#scatter: 산포도. 아규먼트: marker- 검색해서 찾아볼것. x축, y축, s(size)
#바차트: 나중에 보자 이건;
#히스토그램: 나누는 갯수: bins

#당연히, 판다스와 자주 써먹는다. 
#데이터프레임이나 시리즈 별로 사용이 가능하다. 

#EDA란: 탐색적 데이터 분석
#SCATTER MATRIX: 전체적으로 어떤 상관관계를 가지는지 확인하는 용. 근데 컬럼이 많으면 좀 쓸모가 읎다...
