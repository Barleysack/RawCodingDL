import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1) #결과의 재연

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2],[3,1],[4,3],[5,3],[6,2],[3,1],[4,3],[5,3],[6,2],[3,1],[4,3],[5,3],[6,2],[3,1],[4,3],[5,3],[6,2]] #6,2 #m은 6, d는 2
y_data = [[0],[0],[0],[1],[1],[1],[0],[1],[1],[1],[0],[1],[1],[1],[0],[1],[1],[1],[0],[1],[1],[1]]#6,1


x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

#H(x)의 계산.

w = torch.zeros((2,1),requires_grad=True) #그래디언트를 학습하리라.
b = torch.zeros((1),requires_grad=True)
optimy = optim.SGD([w,b],lr=1)

hypo = torch.sigmoid(x_train.matmul(w)+b) #시그모이드함수 구현. 근데 사실 이미 torch.sigmoid로 구현되어있다. 
#이는 다시 말해, p가 x가 1일 확률을 이야기한것.
loss = -(y_train * torch.log(hypo)+(1-y_train)*torch.log(1-hypo))
cost = loss.mean()

# 이 모든걸 F.binary_cross_entropy(hypo,y_train) (bce라고 부른다.)
nb_epo = 10000

for i in range(nb_epo+1):

    hypo = torch.sigmoid(x_train.matmul(w)+b)
    cost = F.binary_cross_entropy(hypo,y_train)

    # cost로 h(x) 학습
    optimy.zero_grad()
    cost.backward() #cost에 역전파 수행. 이제까지 사용된 w와 b의 그래디언트를 구해준다.
    optimy.step()#cost 값을 minimize 되는 방향으로 w와 b를 업데이트한다. 


    if i % 100 == 0:
        print('epoch {:4d}/{} Cost: {:.6f}'.format(i,nb_epo,cost.item()))

#evaluation

hypo = torch.sigmoid(x_train.matmul(w)+b)
print(hypo[:5])

prediction = hypo >= torch.FloatTensor([0.5])
print(prediction[:5])
print(y_train[:5])
correct = prediction.float() == y_train
print(correct)