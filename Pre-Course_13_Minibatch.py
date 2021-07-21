import torch
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
#전체 데이터를 균일하게 나눠서 학습하자...!!
#기존 gd보다 조금 더 거칠게 하강한다. 
class MultivariateLinearRegressionModel(nn.Module):
     def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
     def forward(self, x):
        return self.linear(x)
model = MultivariateLinearRegressionModel()
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
    def __len__(self): #데이터 개수 반환
        return len(self.x_data)
    def __getitem__(self, idx):#인덱스를 받아 입출력으로 받아 텐서로 변환해주다.

        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
dataset = CustomDataset()

dataloader = DataLoader(
 dataset,
 batch_size=2, #통상적으로 2의 제곱수로 사용한다.
 shuffle=True, #매번 데이터 학습 순서를 변환한다. 
)
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.0002)
nb_epochs = 10000
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader): #enum이 인덱스를 주니까...!
        x_train, y_train = samples
 # H(x) 계산
        prediction = model(x_train)
 # cost 계산
        cost = F.mse_loss(prediction, y_train)
 # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
#엥 이거 모델링 좀 이상한거같은데...?