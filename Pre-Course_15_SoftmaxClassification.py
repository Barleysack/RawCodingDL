import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



torch.manual_seed(1)


#이산 확률 분포 : 딱 떨어진 해당 값만 가지고 있을것. 가위바위보라던가..?



z2 = torch.rand(3,5,requires_grad=True)
hypo2 = F.softmax(z2,dim=1)
print(hypo2) #클래스 5개, 샘플 3개. 

y = torch.randint(5,(3,)).long()
y_one_hot = torch.zeros_like(hypo2)
y_one_hot.scatter_(1,y.unsqueeze(1),1)

cost = (y_one_hot* -torch.log(hypo2)).sum(dim=1).mean()
#nll loss: negative log likelihood 
print(F.cross_entropy(z2,y))


class SMclassmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3) #output이 3
    def forward(self, x):
        return self.linear(x)

model =SMclassmodel() #x = m,4 -> m,3
optimizer = optim.SGD(model.parameters(),lr = 0.1)

nb_epochs = 1000

for i in range(nb_epochs+1):
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    