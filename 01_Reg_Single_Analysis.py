import numpy as np
import csv
import time
from numpy.core.fromnumeric import transpose
from numpy.random import shuffle

np.random.seed(7993)

def randomize(): np.random.seed(time.time())

RND_MEAN      = 0
RND_STD       = 0.0030
LEARNING_RATE = 0.001


#난숫값 평균/ 표준편차

#순서: exec->load/init_model/train_and_test->arrange_Data/get_train_data/get_test_data/run_train/run_test
#->forward_neuralnet/forward_postproc/eval_accuracy->backprop_neuralnet->backprop_postproc

#메인함수 정의

def abalone_exec(epoch_count=10,mb_size=10,report=1):
    load_abalone_dataset()                              #데이터셋 적재
    init_model()                                        #모델 파라미터 초기화
    train_and_test(epoch_count,mb_size,report)          #학습 및 평가
#=================================================================================================================
def load_abalone_dataset():    
    with open('./Datasets/abalone.csv') as csvfile:     #csv 모듈로부터 가져옴 
        csvreader = csv.reader(csvfile)    
        next(csvreader,None)                            #파일 첫 행을 읽지 않고 건너뛴다. (헤더행 무시)
        rows = []                                       #각 행의 정보를 담을 rows
        for row in csvreader:   
            rows.append(row)  
    global data,input_cnt,output_cnt  
    input_cnt,output_cnt = 10,1                         #입출력 벡터 크기 지정
    data = np.zeros([len(rows),input_cnt+output_cnt])   #입출력 벡터 정보 저장할 data 행렬 만들때 크기 지정에 이용 
  
    for n, row in enumerate(rows):                      # 비선형인 성별 정보를 원-핫 벡터 표현으로 변환하는 처리
        if row[0] == 'I': data[n,0] =1                  #모름
        if row[0] == 'M': data[n,1] =1                  #수컷
        if row[0] == 'F': data[n,2] =1                  #암컷
        data[n,3:] = row[1:]                            #나머지 정보 항목을 일괄 복제한다.
#=================================================================================================================
  
def init_model():
    global weight,bias, input_cnt, output_cnt 
    weight = np.random.normal(RND_MEAN,RND_STD,[input_cnt,output_cnt])
    #정규 분포를 가지는 난수로 초기화한다. 경사하강법 출발점에서 매번 다르도록 하기 위함이다.
    bias = np.zeros([output_cnt])                       #편향은 0으로 초기화하여 생성.
#=================================================================================================================

def train_and_test(epoch_count,mb_size, report):
    step_count = arrange_data(mb_size)
    test_x , test_y = get_test_data()
    for epoch in range(epoch_count):                    #epoch_count 인수로 지정된 에포크 수만큼 학습을 반복하며 반복문 안에서 다시 
        losses, accs = [],[]

        for n in range(step_count):                     #step_count 값 만큼 미니배치 처리를 반복한다. 이때 step_count 값은 반복문 시작 전에 
                                                        #arrange_data()를 호출해 구하는데, 이것은 데이터 전처리 함수이다. 
            train_x,train_y = get_train_data(mb_size,n) #미니배치 데이터 얻어와 run_train 호출.
            loss, acc = run_train(train_x, train_y)     #이떄 미니배치 단위의 비용과 정확도를 보고받아 리스트 변수 losses와 accs에 집계한다. 
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:      #각 에포크 처리가 끝나면 report 에 지정된 보고 주기에 해당하는지 검사한다. 
        #                                               #해당되면 중간 평가 함수 run_test()를 호출 하고 그 결과를 호출한다. 
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss={:5.3f}, accuracy = {:5.3f}/{:5.3f}'\
                .format( epoch+1,np.mean(losses),np.mean(accs),acc))
            
    final_acc = run_test(test_x,test_y)                 #전체 에포크 처리가 끝나면 다시 최종 평가함수 호출 후 출력. 같은 데이터셋을 사용하므로 루프 바깥에 get_test_data.
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

#=================================================================================================================

def arrange_data(mb_size):                              #이거 아까 그 전처리 함수였지? 
    """얘가 뭔가 문제가 있는 것 같은데... """
    global data 
    global shuffle_map
    global test_begin_idx
    ishape=np.shape(data[0])[0]
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
                                                        #데이터 수 만큼의 일련번호를 발생시킨 후 
                                                        #무작위로 순서를 섞는다. 이 shuffle_map은 get_train_data와 get_test_data에서 사용된다. 
                                              
    test_begin_idx = step_count * mb_size
                                                        #또한 테스트데이터와 검증데이터의 경계를 인덱스로 저장한 후 에포크 학습에 필요한 미니배치 처리 스텝수를 반환한다.  

    return step_count
#=================================================================================================================

def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt #test_begin_idx를 경계로 
                                                         #인덱스 배열 shuffle_map의 후반부를 평가용 데이터로 반환.
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:] #앞쪽을 입력벡터, 뒷쪽을 정답 벡터로서 반환. 
#=================================================================================================================

def get_train_data(mb_size, nth): 
    global data, shuffle_map, test_begin_idx, output_cnt #test data 함수와 비스무리-
                                                         #미니배치 구간의 위치를 따져 그 구간의 데이터만을 shuffle_map에서 반환.
    if nth == 0:                                         #각 epoch 첫 호출에서만 해당 학습 데이터의 순서를 뒤섞어 에폭마다 다른 순서로 학습 수행. 
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:] #역시 반화닛에는 입력과 정답을 분할해 반환.
#=================================================================================================================


def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)                #순전파 처리, 단층 퍼셉트론 신경망 처리, 입력행렬 x로 부터 신경망 출력 output을 구한다.
    loss, aux_pp = forward_postproc(output, y)           #순전파 처리, 회귀 분석에 맞춘 순전파 작업을 수행해 output과 y로부터 손실함수 loss를 계산한다.
    accuracy = eval_accuracy(output, y)                  #보고용 정확도 계산, 보고용 eval_accuracy 호출해 accuracy에 저장. 
                                                         #나눈 이유는 이 두 단계를 독립적으로 분리하여 사용하기 위함이다. 문제가 바뀌면 forward_postproc만 슬쩍슬쩍 바꾼다.
                                                         #신경망 구조만 바꾼다면 forward_neuralnet만 건드리는 식.
    G_loss = 1.0                                         #역전파의 시작점., 로L/로L이기 때문. 
    G_output = backprop_postproc(G_loss, aux_pp)         #역전파 처리 ,순전파의 역순으로 호출. 순전파 출력인 손실 기울기를 입력으로 받아 내부 처리를 마친 후 
                                                         #순전파때 입력이었던 성분의 손실 기울기를 반환. G_loss로부터 G_output을 구한다.
    backprop_neuralnet(G_output, aux_nn)                 #역전파 처리. 순전파때는 첫단계였던 신경망 함수 콜- G_output으로부터 g_x를 구한다. 
                                                         #이때 backprop_neuralnet은 별도의 값을 리턴하지 않아도 되며, 그저 함수가 호출되었을때 파라미터가 학습된다는 것이 중요하다. 
        
    return loss, accuracy
#=================================================================================================================





def run_test(x, y):                                      
    output, _ = forward_neuralnet(x)                     #순전파 처리만 수행하고
    accuracy = eval_accuracy(output, y)                  #바로 정확도를 계산해 반환한다. 학습이 아니잖아?
    return accuracy








#=================================================================================================================






def forward_neuralnet(x):                                
    global weight, bias                                  #행렬끼리의 곱셈과, 편향 덧셈은 행렬/벡터 덧셈(ax+b)
    output = np.matmul(x, weight) + bias                 #10가지 특성, [N,10]이 X가 되고, 가중치는 [10,1]로 곱하면 [N,1]이 된다.
                                                         #단순히 더해지는 bias는 원래라면 모양이 맞지 않으나 파이썬 인터프레터가 
                                                         #알아서 행렬의 각 행에 해당 값을 더한다.
    return output, x                                     #출력과 입력을 다시 반환. 





#=================================================================================================================




def backprop_neuralnet(G_output, x):                     #손실기울기 G_output을 전달받아 
    global weight, bias                     
    g_output_w = np.transpose(x)                         #이때 전달된 x를 통해 x와 output 사이 부분 기울기를 구해냄. 
                                                         #어째 넘파이 버전이 다른 것 같아서 ;; 살짝 바꿔뒀습니다.    
                                                         #g_output_w는 x의 전치니 [10,N]이 되고 G_output은 output과 같은 [N,1]로 전달. 
    G_w = np.matmul(g_output_w, G_output)                #가중치의 손실 기울기 G_w는 그렇기에 [10,1]로 나온다. 이는 weight와 같은 모양이 되네?  
    G_b = np.sum(G_output, axis=0)                       #편향치의 손실 기울기 G_b , 둘다 G_output으로 부터. 이 역시 bias와 동일한 모양이 된다. 

    weight -= LEARNING_RATE * G_w                        #실제 학습의 수행, 위에서 모양 다 맞춰놨지 ! 
    bias -= LEARNING_RATE * G_b                          #실제 학습의 수행, 위에서 모양 다 맞춰놨지 ! 





#=================================================================================================================




def forward_postproc(output, y):                         #단층 퍼셉트론 출력으로부터 손실 함숫값을 구하는 과정. 
    diff = output - y                                    #MSE를 3단계로 수행. 정답행렬 y에 대해 각 행렬 원소 짝에 대한
                                                         #오차와 그 제곱을 diff와 square로 구함. 
    square = np.square(diff)                             #제...곱인데...왜 없지;; 넘파이 버전 문제로 위에서부터 고쳐나가는 중.
    loss = np.mean(square)                               #이들의 평균으로 loss를 구한다. 그 후 이를 반환.
    return loss, diff




#=================================================================================================================



def backprop_postproc(G_loss, diff):                     #순전파 역순으로 G_output을 구해 반환. 
    shape = diff.shape                                   #G_Loss값(초기 1) 로부터 평균, 제곱, 오차 연산에 대한 역전파 처리를 수행.
    g_loss_square = np.ones(shape) / np.prod(shape)      #각 단계간 부분 기울기를 구해두고 이후 손실 기울기의 연쇄적 계산에 사용.
    g_square_diff = 2 * diff                             #각 단계간 부분 기울기를 구해두고 이후 손실 기울기의 연쇄적 계산에 사용.
    g_diff_output = 1                                    #각 단계간 부분 기울기를 구해두고 이후 손실 기울기의 연쇄적 계산에 사용.

    G_square = g_loss_square * G_loss                    #아래 세 식은 모두 항등식이다. 역전파 과정을 이해하기 위해 쪼개놓은 것 뿐.
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff
    
    return G_output





#=================================================================================================================




def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))              #정답과 오차의 비율을 오류율로 보고 1에서 오류율 평균을 뺀 값으로 정확도를 정의.
    return 1 - mdiff



    
#=================================================================================================================



def backprop_postproc_oneline(G_loss, diff):             # backprop_postproc() 대신 사용 가능
    return 2 * diff / np.prod(diff.shape)                #위의 함수를 간단히 만들면 이리 된다. 



#=================================================================================================================


#실행.

abalone_exec()