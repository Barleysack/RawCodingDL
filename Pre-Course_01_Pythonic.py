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
#1+2,3+3,6+4,10+5를 수행.

#iterable한 객체는 iter와 next로 이루어짐. iter로 iterator object로 선언, next로 다음 원소를 가져옴.

#generator: iterable object를 특수한 형태로 사용. yield를 사용해 한번에 하나의 원소만 반환. 
#[]대신 ()를 사용하여 표현
#generator는 iterator에 비해 훨씬 작은 메모리 용량 사용. 최적활르 위해 알아둘것. 
gen_ex = (n*n for n in range(500)) #이게
list_ex = [n*n for n in range(500)] #이것보다 메모리 용량 선에서 효율적.
#list 타입의 데이터를 반환해주는 함수는 generator로 만들어라. 중간 과정에서 loop이 중단될 수 있을때. 
#큰 데이터를 처리할때는 제너레이터가 유용. 파일 데이터를 처리할때도 상당히 유용하다. 


#함수에 입력되는 arguments:
#1.keyword Args
#2.Default Args
#3.Variable Length Args
a = "me"
b = "hehe"
def printy(first,second):
    print("hello {1}, it's {0}".format(b,a))
printy(first="hehe",second="doh")
#파라미터의 변수명을 사용, arguments를 넘김
def printy2(first,second="default"):#디폴트로 인자에 들어갈 값을 정해놓는다.
    print("hello {1}, it's {0}".format(b,a))

#함수의 parameter가 정해지지 않았을때. 다항방정식/마트물건계산?
#개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 법.
#Keyword arguments와 함께, argument 추가가 가능.
# * 기호 활용, 입력된 값은 tuple type!
#가변 인자는 오직 한개 마지막 parameter 위치에 사용 가능.

def aster_test(a,b,*args):
    return a+b+sum(*args)

a = aster_test(1,2,(3,4,5))

#키워드 가변인자. dict type으로 사용 가능하다. 
