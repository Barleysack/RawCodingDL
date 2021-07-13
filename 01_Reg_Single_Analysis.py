import numpy as np
import csv
import time

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
    load_abalone_dataset()                            #데이터셋 적재
    init_model()                                      #모델 파라미터 초기화
    train_and_test(epoch_count,mb_size,report)        #학습 및 평가

def load_abalone_dataset():  
    with open('./Datasets/abalone.csv') as csvfile:   #csv 모듈로부터 가져옴 
        csvreader = csv.reader(csvfile)  
        next(csvreader,None)                          #파일 첫 행을 읽지 않고 건너뛴다. (헤더행 무시)
        rows = []                                     #각 행의 정보를 담을 rows
        for row in csvreader: 
            rows.append(row)
    global data,input_cnt,output_cnt
    input_cnt,output_cnt = 10,1                       #입출력 벡터 크기 지정
    data = np.zeros([len(rows),input_cnt+output_cnt]) #입출력 벡터 정보 저장할 data 행렬 만들때 크기 지정에 이용 

    for n, row in enumerate(rows):                    # 비선형인 성별 정보를 원-핫 벡터 표현으로 변환하는 처리
        if row[0] == 'I': data[n,0] =1                #모름
        if row[0] == 'M': data[n,1] =1                #수컷
        if row[0] == 'F': data[n,2] =1                #암컷
        data[n,3:] = row[1:]                          #나머지 정보 항목을 일괄 복제한다.

def init_model():
    global weight,bias, input_cnt, output_cnt 
    weight = np.random.normal(RND_MEAN,RND_STD,[input_cnt,output_cnt])
    #정규 분포를 가지는 난수로 초기화한다. 경사하강법 출발점에서 매번 다르도록 하기 위함이다.
    bias = np.zeros([output_cnt]) #편향은 0으로 초기화하여 생성.

def train_and_test(epoch_count,mb_size, report):
    step_count = arrange_data(mb_size)
    test_x , test_y = get_test_data()
    for epoch in range(epoch_count): #epoch_count 인수로 지정된 에포크 수만큼 학습을 반복하며 반복문 안에서 다시 
        losses, accs = [],[]

        for n in range(step_count):  #step_count 값 만큼 미니배치 처리를 반복한다. 이때 step_count 값은 반복문 시작 전에 
                                     #arrange_data()를 호출해 구하는데, 이것은 데이터 전처리 함수이다. 
            train_x,train_y = get_train_data(mb_size,n) # 미니배치 데이터 얻어와 run_train 호출.
            loss, acc = run_train(train_x, train_y)     # 이떄 미니배치 단위의 비용과 정확도를 보고받아 리스트 변수 losses와 accs에 집계한다. 
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0: #각 에포크 처리가 끝나면 report 에 지정된 보고 주기에 해당하는지 검사한다. 
        #                                          #해당되면 중간 평가 함수 run_test()를 호출 하고 그 결과를 호출한다. 
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss={:5.3f}, accuracy = {:5.3f}/{:5.3f}'\
                .format( epoch+1,np.mean(losses),np.mean(accs),acc))
            
    final_acc = run_test(test_x,test_y) #전체 에포크 처리가 끝나면 다시 최종 평가함수 호출 후 출력. 같은 데이터셋을 사용하므로 루프 바깥에 get_test_data.
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))


def arrange_data(mb_size): #이거 아까 그 전처리 함수였지? 
    global data, shuffle_map, test_begin_idx 
    shuffle_map = np.arange(data.shape[0]) #데이터 수 만큼의 일련번호를 발생시킨 후 
    np.random.shuffle(shuffle_map) #무작위로 순서를 섞는다. 이 shuffle_map은 get_train_data와 get_test_data에서 사용된다. 
    step_count = int(data.shape[0] * 0.8) // mb_size 
    test_begin_idx = step_count * mb_size
                                   #또한 테스트데이터와 검증데이터의 경계를 인덱스로 저장한 후 에포크 학습에 필요한 미니배치 처리 스텝수를 반환한다.  

    return step_count

def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt #test_begin_idx를 경계로 
                                                         #인덱스 배열 shuffle_map의 후반부를 평가용 데이터로 반환.
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:] #앞쪽을 입력벡터, 뒷쪽을 정답 벡터로서 반환. 

def get_train_data(mb_size, nth): 
    global data, shuffle_map, test_begin_idx, output_cnt #test data 함수와 비스무리-
                                                         #미니배치 구간의 위치를 따져 그 구간의 데이터만을 shuffle_map에서 반환.
    if nth == 0:                                         #각 epoch 첫 호출에서만 해당 학습 데이터의 순서를 뒤섞어 에폭마다 다른 순서로 학습 수행. 
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:] #역시 반화닛에는 입력과 정답을 분할해 반환.


def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)        # 순전파 처리, 단층 퍼셉트론 신경망 처리, 입력행렬 x로 부터 신경망 출력 output을 구한다.
    loss, aux_pp = forward_postproc(output, y)   #순전파 처리, 회귀 분석에 맞춘 순전파 작업을 수행해 output과 y로부터 손실함수 loss를 계산한다.
    accuracy = eval_accuracy(output, y)          #보고용 정확도 계산, 보고용 eval_accuracy 호출해 accuracy에 저장. 
                                                 #나눈 이유는 이 두 단계를 독립적으로 분리하여 사용하기 위함이다. 문제가 바뀌면 forward_postproc만 슬쩍슬쩍 바꾼다.
                                                 #신경망 구조만 바꾼다면 forward_neuralnet만 건드리는 식.
    G_loss = 1.0                                 #역전파의 시작점., 로L/로L이기 때문. 
    G_output = backprop_postproc(G_loss, aux_pp) #역전파 처리 ,순전파의 역순으로 호출. 순전파 출력인 손실 기울기를 입력으로 받아 내부 처리를 마친 후 순전파때 입력이었던 성분의 손실 기울기를 반환.
    backprop_neuralnet(G_output, aux_nn)         #역전파 처리. 
    
    return loss, accuracy

def run_test(x, y): 
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy

def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x

def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()
    
    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b



def forward_postproc(output, y):
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff

def backprop_postproc(G_loss, diff):
    shape = diff.shape
    
    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff
    
    return G_output


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))
    return 1 - mdiff
def backprop_postproc_oneline(G_loss, diff):  # backprop_postproc() 대신 사용 가능
    return 2 * diff / np.prod(diff.shape)