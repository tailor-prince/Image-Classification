#[batch,channel,height,width]
# N=(W-F+2P)/S+1
# N:输出图片大小
# W:输入图片大小为 W*W
# F:卷积核为 F*F
# P:Paddding
# S:步长

import torch.nn as nn
import torch.nn.functional as F
import torch

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1=nn.Conv2d(3,16,5) #in_channels,out_channels,kernel_size  （32-5+0）/1+1=28
        self.pool1=nn.MaxPool2d(2,2) #kernel_size
        self.conv2=nn.Conv2d(16,32,5)
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=F.relu(self.conv1(x)) #input(3,32,32)
        x=self.pool1(x)         #output(16,14,14)
        x=F.relu(self.conv2(x)) #output(32,10,10)
        x=self.pool2(x)        #output(32,5,5)
        x=x.view(-1,32*5*5)     #output(32*5*5)
        x=F.relu(self.fc1(x))   #output(120)
        x = F.relu(self.fc2(x)) #output(84)
        x = self.fc3(x)         #output(10)
        return x

def main():
    device = torch.device('cuda')
    input = torch.rand([32, 3, 32, 32]).to(device)
    model = Lenet().to(device)
    print(model)

    print(model(input).shape)

if __name__ == '__main__':
    main()
