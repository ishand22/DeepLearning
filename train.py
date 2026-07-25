import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from Models import MLP
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
train_dataset=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform # convert PIL to tensor and normalize the pixels in 0,1
)
loader =DataLoader(
    train_dataset,batch_size=64, shuffle=True
)
image,label=train_dataset[0]
print(image,label)
print(type(image))
print(type(label))
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,128)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        #print(x.shape)
        x=x.view(x.size(0),-1)
        #print(x.shape)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
model=MLP()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
criterion=nn.CrossEntropyLoss()
for epoch in range(5):
    running_loss=0
    for image,label in loader:
        output=model(image)
        loss=criterion(output,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f"Epoch{epoch+1}: Loss={running_loss/len(loader):.4f}")



test_dataset=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
test_loader=DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
model.eval()
correct=0
total=0
with torch.no_grad():
    for image,label in test_loader:
        output=model(image)
        predicted=output.argmax(dim=1)
        correct+=(predicted==label).sum().item()
        total+=label.size(0)
accuracy=100*correct/total
print(f"{accuracy}")
