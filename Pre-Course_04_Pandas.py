import pandas as pd
from pandas.core.base import NoNewAttributesMixin
from pandas.core.series import Series
import numpy as np


data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df_data = pd.read_csv(data_url,sep='\s+',header = None) #\s면 빈칸으로 나누어 다 가져오라.
print(df_data.head()) #데이터프레임이네. #Series:Column Vector를 표현하는 object
print(df_data.values)
list_data = [1,2,3,4,5]
list_datay= Series (list_data, dtype=np.float32, name ="examply_data")

#데이터프레임은 인덱스와 컬럼의 값을 이용해 찾는다. 넘파이 기능 대부분 사용 가능 .
#판다스에선 컬럼의 이름을 지정하고, 해당 컬럼에 맞는 데이터를 뽑아올 수 있다.
#새로운 컬럼의 추가 또한 가능하다. 

"""어려운 부분? : .loc: 인덱스의 이름을 가지고 특정 위치의 값을 접근한다. ilock 행 번호를 기준으로 행 데이터 읽기. """

"""T,values,to_csv등의 함수 또한 사용 가능. """

"""컬럼 삭제: del로 합시다. 각 컬럼이 object인 연유. """

"""여러개의 컬럼: 대괄호로 꼭 표시해야 한다."""

"""head: 상단부터... 컬럼 지정시 시리즈 데이터로서 가져온다. 아니라면 그냥 인덱스 넘버 해당하는 것을 가져온다."""

"""(헷갈리는 문법이니 지양)대괄호 내에 대괄호로서 나타낼 시 여러개의 인덱스또한 지정이 가능하다."""

"""인덱스 변경: PK 생각하면서 해보셔도 무방하지 않겠나.."""

"""Column과 index number.그냥 df.index하는 식도 양호. """

"""iloc을 사용하면 숫자로만 편안히 하는걸 추천. 컬럼이 많을때는 아무래도 name으로 하게된다."""

"""row 단위 data를 없앨때 index number로 drop 사용. 그냥 drop을 하면 복사본만 사라지는것..."""

"""drop을 이용, column을 없애고 싶으면 그 이름과 axis를 1로 하여 날려라. axis 관련해서 이후 잠깐 체크해보는 것도 좋을듯."""

"""원본데이터를 변환시켜주고싶으면 inplace 아규먼트 지정. inplace=true면 데이터프레임에 적용."""

"""add 연산은 nan을 0으로 변환."""

"""column을 기준, broadcasting 발생."""


"""시리즈 객체에도 map과 람다가 상당히 편안히 이용될 수 있다."""
s = Series(np.arange(10))
z ={1:'D',2:'E'}
s.map(z)
print(s)
#뭐 이런식으로...dict타입으로 자주 사용하는 편이다. 
#replace 또한 잘 써먹는다.  -이것도 inplace 속성이 있다.

#apply : map과 달리 시리즈 전체(column)에 해당 함수를 적용.
#입력값도 series 데이터로 입력받아 핸들링기능. 통계를 쓸때 자주 한다.
#데이터프레임 내 통계자료 또한 편안히 사용할 수 있다. 
#applymap: 시리즈단위가 아닌 element 단위로 함수를 적용함.
#series 단위에 apply를 적용할 때와 같은 효과.

#판다스 내장함수
#데이터프레임: describe, unique, 
#각자 카테고리에 뭐가 있나 싶을때 enum/dict/unique로 처리 가능.
 



