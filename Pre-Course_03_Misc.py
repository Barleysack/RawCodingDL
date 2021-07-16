#Exception handling

a = [1,2,3,4,5]
for i in range(1,10):
    try:
        print(i,10//i)
    except ZeroDivisionError: #ZeroDivisionError
        print("Error")
        print("Not Divided by zero.")
    except Exception:
        print(Exception) #전체로 잡는건 그리 좋지 못한 선택.
    finally:
        print("=========================")
    if i == 5:
        raise Exception("멈춰...!") #예외를 일으킨다. 의도적으로.
    assert isinstance(i%2==0,int)


#with 구문도 익혀두는게 좋지 않을까 생각. 
    
with open("파일명".txt,"r") as my_file:
    i=0
    while True:
        line = my_file.readline() #한줄씩 값 출력
        if not line:
            break
        print(str(i)+"===" + line.replace("\n",""))
        i+=1
#mode가 w라면 write. 인코딩은 보통 utf8.
if not os.path.isdir("log"):
    os.mkdir(log.log)


#pickle 파이썬의 객체를 영속화하는 built-in 객체
#data, object 등 실행중 정보를 저장 , 불러와서 사용
#저장해야하는 정보, 계산결과(모델) 등 활옹이 많음

import pickle #pickle은 파이썬에 특화된 바이너리 파일. 


f = open("list.pickle", "wb")
test = [1, 2, 3, 4, 5]





pickle.dump(test, f)
f.close()
f = open("list.pickle", "rb")
test_pickle = pickle.load(f)
print(test_pickle)
f.close()

class Mutltiply(object):
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def multiply(self, number):
        return number * self.multiplier


muliply = Mutltiply(5)
muliply.multiply(10)

f = open("multiply_object.pickle", "wb")
#바이너리에 쓰고,
pickle.dump(muliply, f)

f.close()

f = open("multiply_object.pickle", "rb")
#바이너리파일을 읽는다. 
multiply_pickle = pickle.load(f)

multiply_pickle.multiply(5)

del muliply
#메모리에서 해당 객체를 날려도-
f = open("multiply_object.pickle","rb")
multiply_pickle=pickle.load(f)
multiply_pickle.multiply(100)

#logging : 
#           debug/info/warning/error/critical

import logging
logger = logging.getLogger("main")
stream_hander = logging.StreamHandler()
logger.addHandler(stream_hander)

logger.setLevel(logging.DEBUG)
#debug,info,warning,error,critical의 수준으로 나누어 지정이 가능하다.
#본격적으로 
logger.debug("틀렸잖아!")
logger.info("확인해")
logger.warning("조심해!")
logger.error("에러났어!!!")
logger.critical("망했다...")

#설정의 저장, configparser(파일에). or argparser(실행 시점에).
import configparser
config = configparser.ConfigParser()
config.sections()
config.read('example.cfg')
config.sections()
for key in config['SectionOne']:
    print(key)
config['SectionOne']["status"] 
#키밸류쌍으로 저장할 수 있다. 

#argparser는 진짜 argument들 목록을 사전에 만들어둘 수 있게 하네. 편해보인다. 검색:argparse

