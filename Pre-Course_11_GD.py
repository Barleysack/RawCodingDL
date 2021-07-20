import torch
from torch import optim 


w = torch.zeros(1,requires_grad=True)
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])
hypo = x_train *w #모델
lr = 0.1
#Cost function은 모델의 값이 실제와 얼마나 차이가 있는지 확인하는 함수.
#지금은 mse를 사용할것.
epo = 10
for i in range(epo+1):
    cost = torch.mean((hypo-y_train)**2)
    grad = torch.sum((w*x_train-y_train)*x_train)
    print('Epoch {:4d}/{} W:  {:.3f}, Cost : {:.6f}'.format(i,epo,w.item(),cost.item()))
    w = w - lr*grad
    
#gd를 위한 torch.optim
opt = optim.SGD([w],lr=0.15)
opt.zero_grad() #그래디언트 모두 0으로 초기화.
cost.backward() #cost 미분, 각 변수의 gradient 채움.
opt.step() #저장된 gradient 값으로 gd 시행


#하나의 정보로부터 추측하는 모델일 뿐.
#하지만 여러 정보로부터 추합된 추론이 정확하리라. 

