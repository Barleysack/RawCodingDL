import datetime
import numpy as np


for line in open("./ggl.txt","r"):
    print(line.split()[0])

c = [[1,2,3],
     [4,5,6],
     [7,8,9]]



a = np.array([[2,1],[1,2]])
b = np.array([[2,1],[2,2]])

c = np.dot(a, b) #행렬곱을 위한 닷프로세스.
print(c)