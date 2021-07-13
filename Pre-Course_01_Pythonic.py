#join

colors = ['ab','cd','ef']
q=''.join(colors)
#abcdef


#list comprehension
ex = [x for x in range(10)]
ex2 = [x for x in range(10) if x % 2 == 0]
ex3 = [i+j for i in ex for j in ex2]
ex4 = [i+j for i in ex for j in ex2 if i % 2 == 0 and j % 2 == 0]


# print(ex)
# print(ex2)
# print(ex3) 
# print(ex4)
exk=ex3.sort()

#return은 nonetype이네
#split:문장을 ()안의 기준으로 나눠 list로 변환.
#map:list 객체의 각 요소에 해당하는 함수 적용 후 출력
#enumerate: list의 원소를 추출하며- 번호를 붙여 추출해냄.인덱스와 밸류로서 지정이 된다.

#zip. 두개의 list의 값을 병렬적으로 추출. 

ex2 = ex2 *2



#lambda. 짤막한 함수질.
#함수객체:lambda 인자:리턴값
#map 안에도 들어간다. 다만 이럴거라면 list로 감싸야.
#from functools import reduce
from functools import reduce

print(reduce(lambda x, y : x+y , [1,2,3,4,5]))


