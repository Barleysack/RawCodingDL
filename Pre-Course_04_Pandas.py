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
print(list_datay)