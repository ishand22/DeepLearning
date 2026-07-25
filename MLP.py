import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,128)
        self.relu=nn.ReLU
        self.fc2==nn.Linear(128,10)
    def forward(self,x):
        print(x.shape)
        x=x.view(x.size(0),-1)
        print(x.shape)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x

