import torch,torch.optim
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
#복수의 정보로 하나의 추측값을 가져오자.
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]]) #3회의 쪽지시험 점수
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]) #기말

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)
class MultivariateLinearRegressionModel(nn.Module):
     def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
     def forward(self, x):
        return self.linear(x)
model = MultivariateLinearRegressionModel()





hypothesis = model(x_train)
cost = F.mse_loss(hypothesis,y_train)


nb_epochs = 20
for epoch in range(nb_epochs + 1):
    cost = F.mse_loss(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

 # H(x) 계산
    #  hypothesis = x_train.matmul(W) + b # or .mm or @
    #  # cost 계산
    #  cost = torch.mean((hypothesis - y_train) ** 2)
    #  # cost로 H(x) 개선
    #  optimizer.zero_grad()
    #  cost.backward()
    #  optimizer.step()
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
    epoch, nb_epochs, hypothesis.squeeze().detach(),
    cost.item()))
    #  ))

#multivariable linear regression.

