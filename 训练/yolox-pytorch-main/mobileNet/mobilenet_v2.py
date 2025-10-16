from torch import nn
from torchvision.models.utils import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    # 返回离divisor倍数最近的值，比如v=9,divisor=8,则返回16;v=7,divisor=8,则返回8
    if min_value is None:
        min_value = divisor
        # //: 除法，去掉小数部分
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            # groups=1,普通卷积， groups = in_planes,分组卷积
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

# Inverted resblock:逆残差结构
# 相比于一般的残差结构是先降维再升维
# 而逆残差结构是先升维再降维
# expand_ratio：升维的倍数
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        # boolean
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 如果expand_ratio > 1, 则要通过1*1卷积增加通道数
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # 分组卷积
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # 普通卷积降维
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        # 整个mobilenet-v2
        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        # 残差连接
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            # 每个bottleneck块的参数，每个bottleneck块由n个InvertedResidual组成
            # t:expand_ratio，升维的倍数
            # c:输出通道数
            # n: InvertedResidual的个数
            # s: stride,只针对每个bottleneck块的第一个InvertedResidual，用于将特征图大小变为原来的1/2
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # input_channel:第一层卷积的输出通道数 [h,w,3] -> [h,w,input_channel]
        # _make_divisible调整给定通道数变为8的倍数
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        # 最后一层卷积的输出通道数调整为8的倍数
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:

            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # 只对第一个InvertedResidual
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        # 分类头， fc: 1280->class_num
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层初始化：kaiming_normal_
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def mobilenet_v2(pretrained=False, progress=True):
    model = MobileNetV2()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], model_dir="model_data",
                                              progress=progress)
        model.load_state_dict(state_dict)

    return model

if __name__ == "__main__":
    print(mobilenet_v2())
