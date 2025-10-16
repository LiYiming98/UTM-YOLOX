import torch
import torch.nn as nn

class Gc_block(nn.Module):
    def __init__(self,
                 input_channels,
                 ratio,
                 fusion_type = ('channel_add',)):
        super(Gc_block,self).__init__()
        valid_fusion_types = ['channel_add','channel_mul']
        assert all(f in valid_fusion_types for f in fusion_type),'fusion_type 无效'
        self.conv_wk = nn.Conv2d(input_channels,1,kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.planes = int(input_channels * ratio)

        if 'channel_add' in fusion_type:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(input_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True), # 进行原地操作,x=x+5称为原地操作,y=x+5不是原地操作
                nn.Conv2d(self.planes, input_channels, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_type:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(input_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes,1,1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, input_channels, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self,x):
        b,c,h,w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w)  # 相乘的支路 [b,c,h*w]
        conv_x = self.conv_wk(x) # [b,1,h,w]
        conv_x = conv_x.view(b, 1, h*w)
        conv_x = self.softmax(conv_x) # [b,1,h*w]
        # 由于进行了平铺的操作，维度减少了一维
        input_x = input_x.unsqueeze(1) # [b,1,c,h*w]
        conv_x = conv_x.unsqueeze(-1)  # 最后要进行矩阵相乘，相乘的结果应该是维度为[b,c,1,1], 可由 [b,1,c,h*w] mul [b,1,h*w,1] 得到 [b,1,c,1],再view一下得到
        context_modeling = torch.matmul(input_x,conv_x).view(b,c,1,1)

        return context_modeling

    def forward(self,x):
        context = self.spatial_pool(x)
        ##SENet###################################
        out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = channel_add_term + x

        if self.channel_mul_conv is not None:
            channel_mul_term = nn.Sigmoid(self.channel_mul_conv(context))
            out = channel_mul_term * x

        return out


if __name__ == "__main__":
    module = Gc_block(input_channels=16,ratio=0.25)
    print(module)
    x = torch.randn([1,16,64,64])
    out = module(x)
    print(out.shape)