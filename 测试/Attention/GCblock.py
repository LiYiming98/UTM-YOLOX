import torch
import torch.nn as nn
import torchvision


class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul',)):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        # [f in valid_fusion_types for f in fusion_types]:
        # 等价于 :
        #       result = []
        #       for f in fusion_types:
        #           result.append(f in valid_fusion_types)
        # all 函数: 用于判断给定的可迭代参数iterable中的所有元素是否都为True,否则返回false
        # 元素除了是0，空，false,None外都算True
        # 因此assert all([f in valid_fusion_types for f in fusion_types])的含义是传入的fusion_types必须是valid_fusion_types中的元素
        assert all([f in valid_fusion_types for f in fusion_types]),'传入的fusion_type无效'
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes  # 输入通道数
        self.ratio = ratio # SeNet模块的通道缩放率
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)


        ##########################################################
        # Gcblock 中的Transform模块,类似SeNet,先将通道数降为C/r,再恢复通道数为C
        # 所谓的fusion_types是最后将注意力分数和原始输入融合时的融合方式，可以是相加，也可以是相乘
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        ########################################################

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x) # Wk,将通道数通过卷积降为1
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width) # 将h*w平铺
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            # 如果融合方式是和原始输入相乘，那么要用sigmoid函数归一化
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


if __name__=='__main__':
    model = GlobalContextBlock(inplanes=16, ratio=0.25)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)
