import torch
from torch import optim


x_train =torch.FloatTensor([[1],[2],[3]])
y_train =torch.FloatTensor([[2],[4],[6]])

#Model:Hypothesis
#가장 잘 맞는 하나의 직선.

w = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
hypothesis = x_train*w+b

#우리의 모델이 얼마나 정답과 가까운지.=> cost(w,b)
#mean square error를 사용.

cost = torch.mean((hypothesis-y_train)**2)

optimizer = optim.SGD([w,b],lr=0.05) #학습 대상 및 학습률

epo = 10000

for i in range(1,epo+1):
    hypothesis = x_train*w +b
    cost = torch.mean((hypothesis - y_train)**2)
    optimizer.zero_grad()#그래디언트 초기화.
    cost.backward()#그래디언트 계산
    optimizer.step()#개선
    print("x*{}+{}".format('%.2f'%w.item(),'%.2f'%b.item()))
    if cost <= 0.000001:
        print("the answer is y={}x".format('%.2f'%w.item()))
        break;
print("You should test now! Enter X!")
a = int(input())

c= a*w+b
print("The answer should be {}!".format('%.2f'%c.item()))