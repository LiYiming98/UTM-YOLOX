import torch
from torch import nn

"""SeNet:通道注意力模块
    原理：
    1、通过全局平均池化，将特征图的w,h尺度变为1,1,因此只剩下通道维度
    2、再经过两个全连接层，第一个全连接层将通道维减少为原来的1/16，第二个全连接层将通道数恢复为原来的大小
    3、最后将结果与原始的特征图逐通道维相乘
"""
class SeNet(nn.Module):
    def __init__(self,input_channels,ratio=16):
        super(SeNet,self).__init__()
        # [batch_size,channel,h,w] -> [batch_size,channel,1,1]
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # 因为不用加bias,所以参数为False
            nn.Linear(input_channels,input_channels // ratio,False),
            nn.ReLU(),
            nn.Linear(input_channels // ratio,input_channels,False),
            # 将值映射到[0,1]之间
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,h,w = x.size()
        # [b,c,h,w] -> [b,c,1,1] 然后要接全连接，全连接维度为2，故要将[b,c,1,1] -> [b,c]
        avg = self.avg_pooling(x).view([b,c])
        # [b,c] -> [b,c//16] -> [b,c] -> [b,c,1,1]
        fc = self.fc(avg).view([b,c,1,1])
        return fc * x


input = torch.ones([2,512,20,20])
model = SeNet(512)
print(model)
out_put = model(input)
print(out_put.shape)
