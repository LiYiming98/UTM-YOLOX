import torch.nn as nn
import math
import torch

from nets.darknet import Focus


def _make_divisible(v, divisor, min_value=None): # v:16;divisor:8;min_value:8
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # 16
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """
        h_swish = x * (ReLU6(x+3)/6)
    """
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        # h_sigmoid = ReLU6(x+3)/6
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    """
    SeLayer:
            1、全局平局池化
            2、fc 降低通道数，降低的倍数：reduction
            3、fc 恢复通道数
            4、激活
    """
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    # 3*3卷积，区别是激活为h_swish
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    # 1*1卷积,目的是在3*3卷积后将通道数变为给定的输出通道数
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    """
    Inverted resblock:
    分为两部分：
    (1) 主干部分：首先利用1*1卷积进行升维，然后利用3*3深度可分离卷积进行特征提取，然后再利用1*1卷积降维
    (2) 残差边部分：输入和输出相接
    """
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        # 只有输入通道等于输出通道，且stride=1时才有残差结构
        self.identity = stride == 1 and inp == oup
        # inp==hidden_dim 说明不是botteneck的第一个InvertedResidual,因此先进行一般卷积提取特征
        # 再通过SeLayer调整通道权重，最后1*1卷积将通道降维
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # use_hs为true时使用h_swish,反之使用Relu
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                # nn.Identity(),输入输出不变
                # 当use_se为true时才加入SeLayer,否则不加
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        # 第一个InvertedResidual，要先进行通道升维
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                #
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            #k, t, c, SE, HS, s
            # k: 卷积核大小
            # t: 升维的膨胀率,通过exp_size = _make_divisible(input_channel * t, 8)计算出升维后的通道数
            # c: 输出的通道数，但也通过output_channel = _make_divisible(c * width_mult, 8)计算
            # SE: 是否使用注意力模块
            # HS: 是否使用h_swish激活
            # s: stride
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.6, 24, 0, 0, 1],  # c3: 0-3
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],  # c4 : 4 - 8
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1], # c5 : 9 - 11
            ]



        act="silu"

        input_channel = _make_divisible(16 * width_mult, 8)

        layers = [conv_3x3_bn(3, input_channel, 2)] # input_channel : 16
        # 第一个 conv_3x3_bn(3, input_channel, 2) 层，作用：3*3卷积将输入图尺寸降为原来的一半

        #  InvertedResidual参数： inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs
        #  含义： inp-输入通道数;hidden_dim-升维后的通道数;oup-输出通道数;kernel_size-卷积核大小;
        #  stride:步长;use_se:是否使用通道注意力;use_hs:是否使用h_swish激活
        #
        block = InvertedResidual

        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel # 到下一层
        self.features = nn.Sequential(*layers) # 通过nn.Sequential构建整个模型的特征提取部分
        # 之后的是输出模块，用于分类，这里不需使用
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = _make_divisible(1024 * width_mult, 8) if width_mult > 1.0 else 1024
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenet_v3_small(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('./model_data/mobilenetv3-large-1cd25616.pth')
        model.load_state_dict(state_dict, strict=True)
    return model


# class MobileNetV3_small(nn.Module):
#     def __init__(self, pretrained = False):
#         super(MobileNetV3_small, self).__init__()
#         self.model = mobilenet_v3_small(pretrained=pretrained)
#
#     def forward(self, x):
#         out3 = self.model.features[:4](x)
#         out4 = self.model.features[4:9](out3)
#         out5 = self.model.features[9:12](out4)
#         return out3, out4, out5
#
# if __name__ == "__main__":
#     x = torch.rand([4,3,320,320])
#     module = MobileNetV3_small(pretrained=False)
#     outs = module.forward(x)
#     print(outs[0].shape)
#     print(outs[1].shape)
#     print(outs[2].shape)