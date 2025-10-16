import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# 可分离卷积分为两部分，深度卷积和逐点卷积
# 深度卷积: 分组卷积 + bn + Relu6
#       ——分组卷积: 设置Conv2d的groups参数为输入通道数，意思是将输入按通道分组，然后卷积核也按通道分组
#       假设输入为 20*20*3(inp=3), 一般卷积的卷积核应该为 3*3*3,然后根据输出通道数设置卷积核的个数为oup
#       但对分组卷积，卷积核变为3*3*1，同时输入也按通道分组为3个20*20*1，然后每个3*3*1和20*20*1卷积，输出20*20*1(padding为1)
#       共进行了3次卷积，最终输出为20*20*3,输出的通道数和输入相同。
# 逐点卷积: 1*1卷积 + bn + Relu6
# 逐点卷积是为了将通道数变为oup,操作为一般卷积的操作
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,32
            # 下采样倍数2
            conv_bn(3, 32, 2),
            # 320,320,32 -> 320,320,64
            conv_dw(32, 64, 1),

            # 320,320,64 -> 160,160,128
            # 下采样倍数4
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 下采样倍数8
            # 160,160,128 -> 80,80,256
            conv_dw(128, 256, 2),

            conv_dw(256, 256, 1),
        )
        # 80,80,256 -> 40,40,512
        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        # 40,40,512 -> 20,20,1024
        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # 分类头
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def mobilenet_v1(pretrained=False, progress=True):
    model = MobileNetV1()
    if pretrained:
        print("mobilenet_v1 has no pretrained model")
    return model


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v1().to(device)
    summary(model, input_size=(3, 416, 416))
