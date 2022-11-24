import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
sequence_length = 28 # 행을 시계열로 보겠다.
num_layers = 1
hidden_size = 128

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True
        ) # B, S, F
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x) -> torch.Tensor:
        h0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size)) # size(0) == batch_num

        # forward prop
        out, _ = self.rnn(x, h0)
        # out = out.reshape(out.shape[0], -1) # 다 나온 행들을 하나로 합침. 3차원 -> 2차원
        out = self.fc(out[:, -1, :]) # 배치, 마지막 시퀀스, 피처 히든사이즈 -> 64, 1, 128 이 아니라 64, 128 됨.
        return out
    

train_dataset = datasets.MNIST(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # data = data.view(data.size(0), -1)
        # 이미지를 한줄로 폄. -> 이것이 시계열이 됨
        # mlp 할때 했던것. 이제 그대로 넣어서 행이 시계열이 됨.

        scores = model(data)
        loss = criterion(scores, targets)

        # loss backward 시 모든 그라디언트 계산
        optimizer.zero_grad()
        loss.backward()

        # 계산된 그라디언트를 러닝레잍 곱하고 빼는 과정
        optimizer.step()

        
def check_accuracy(loader: DataLoader, model: nn.Module):
    if loader.dataset.train:
        print('checking acc on train data')
    else:
        print('checking acc on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1) # 원래 데이터셋 형태가 이럼
            y = y.to(device)

            scores = model(x)
            _, preds = scores.max(1) # 값, 인덱스!
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        print(f'Got {num_correct} / {num_samples} with acc \
            {float(num_correct)/float(num_samples)}')
        
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


