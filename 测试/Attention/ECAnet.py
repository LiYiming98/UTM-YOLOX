import torch
from torch import nn
import math
class ECAnet(nn.Module):
    """ECANet是对SeNet的改进，
    它的作者认为SeNet中第一个全连接层对通道维
    实施降维给通道注意预测带来了副作用，
    而且捕获所有通道之间的依赖是低效的，也是不必要的。
    因此ECANet就是将SeNet的两个全连接用一个1*1的卷积层代替，这样就不用对通道维降维
    """
    def __init__(self,input_channel,gamma=2,b=1):
        super(ECAnet,self).__init__()
        kernel_size = int(abs((math.log(input_channel,2)+b)/gamma))
        kernel_size = kernel_size if kernel_size %2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size,padding=kernel_size //2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        # 将全局平均池化后的tensor变成序列形式
        # batch_size维度可以不看
        output = self.avg_pooling(x).view([b,1,c])
        output = self.conv(output)
        output = self.sigmoid(output).view([b,c,1,1])
        print(output)
        return output * x

x = torch.ones([2,512,26,26])
module = ECAnet(512)
print(module)
output = module(x)
print(output)