from collections import OrderedDict

import torch
import torch.nn as nn

from mobileNet.mobvilenet_v3_small import MobileNetV3 as mobileNetv3
from nets.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from Attention.Cbam import cbam_block
from utils.utils import get_classes
###########################################################################
def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))

def conv_dw(filter_in, filter_out, stride = 1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


class MobileNetV3(nn.Module):
    def __init__(self, pretrained = False):
        super(MobileNetV3, self).__init__()
        self.model = mobileNetv3()

    def forward(self, x):
        out3 = self.model.features[:4](x)
        out4 = self.model.features[4:9](out3)
        out5 = self.model.features[9:12](out4)
        return out3, out4, out5

def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        conv_dw(filters_list[0], filters_list[1]),
        conv_dw(filters_list[1], filters_list[0]),

    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        conv_dw(filters_list[0], filters_list[1]),
        conv_dw(filters_list[1], filters_list[0]),
        conv_dw(filters_list[0], filters_list[1]),
        conv_dw(filters_list[1], filters_list[0]),
    )
    return m

###########################################################################
class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[128, 256, 512], act="silu", depthwise=False,reg_max = 14):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(128 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                # cbam_block(int(128 * width)),
                Conv(in_channels=int(128 * width), out_channels=int(128 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(128 * width), out_channels=int(128 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(128 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(128 * width), out_channels=int(128 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(128 * width), out_channels=int(128 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(128 * width), out_channels=4 * (reg_max + 1), kernel_size=1, stride=1,
                          padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(128 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )


    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)

            outputs.append(output)
        return outputs


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[128, 256, 512],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        # self.backbone:<"dark3",output1>,<"dark4",output2>,<"dark5",output3>
        # self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        #########################################################################
        self.backbone = MobileNetV3(pretrained=False)    # 加载MobileNetV3 backbone
        alpha = 1
        in_filters = [24, 48, 96]

        # 从backbone得到的三个级别的特征图： [10,10,96] , [20,20,48] , [40,40,24],分别经过三个卷积模块调整通道数
        # [10,10,96] —> [10,10,256]
        self.conv1 = make_three_conv([int(256 * alpha), int(512 * alpha)], in_filters[2])
        # [20,20,48] -> [20,20,128]
        self.make_three_conv1 = make_three_conv([int(128 * alpha), int(256 * alpha)], in_filters[1])
        # [40,40,24] -> [40,40,64]
        self.make_three_conv2 = make_three_conv([int(64 * alpha), int(128 * alpha)], in_filters[0])


        # PAN
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # -------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        # -------------------------------------------#
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        # -------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        # -------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)

        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)

        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # yolox-s
        # self.feat_attention1 = cbam_block(128)
        # self.feat_attention2 = cbam_block(256)
        # self.feat_attention3 = cbam_block(512)
        # self.feat_attention4 = cbam_block(256)
        # self.feat_attention5 = cbam_block(128)

        # yolox-x cbam
        # self.feat_attention1 = cbam_block(320)
        # self.feat_attention2 = cbam_block(640)
        # self.feat_attention3 = cbam_block(1280)
        # self.feat_attention4 = cbam_block(640)
        # self.feat_attention5 = cbam_block(320)

        # yolox-s-mobilenet
        # self.feat_attention1 = cbam_block(64)
        # self.feat_attention2 = cbam_block(128)
        # self.feat_attention3 = cbam_block(256)
        # self.feat_attention4 = cbam_block(128)
        # self.feat_attention5 = cbam_block(64)

    def forward(self, input):
        # out1: [40,40,24] , out2: [20,20,48] , out3: [10,10,96]
        out1,out2,out3 = self.backbone.forward(input)
        # feat1 : [40,40,64] feat2 : [20,20,128] , feat3 : [10,10,256]
        feat1 = self.make_three_conv2(out1)
        feat2 = self.make_three_conv1(out2)
        feat3 = self.conv1(out3)

        # [10,10,256] -> [10,10,128]
        P5 = self.lateral_conv0(feat3)

        # [10,10,128] -> [20,20,128]
        P5_upsample = self.upsample(P5)
        # 上采样后加一层注意力
        #P5_upsample = self.feat_attention4(P5_upsample)

        # [20,20,128] + [20,20,128] -> [20,20,256]
        P5_upsample = torch.cat([P5_upsample, feat2], 1)

        # [20,20,256] -> [20,20,128]
        P5_upsample = self.C3_p4(P5_upsample)

        # [20,20,128] -> [20,20,64]
        P4 = self.reduce_conv1(P5_upsample)

        # [20,20,64] -> [40,40,64]
        P4_upsample = self.upsample(P4)
        # 加注意力
        #P4_upsample = self.feat_attention5(P4_upsample)

        # [40,40,64] + [40,40,64] -> [40,40,128]
        P4_upsample = torch.cat([P4_upsample, feat1], 1)

        # [40,40,128] -> [40,40,64]
        P3_out = self.C3_p3(P4_upsample)

        # [40,40,64] -> [20,20,64]
        P3_downsample = self.bu_conv2(P3_out)

        # [20,20,64] -> [20,20,128]
        P3_downsample = torch.cat([P3_downsample, P4], 1)

        # [20,20,128] -> [20,20,128]
        P4_out = self.C3_n3(P3_downsample)

        # [20,20,128] -> [10,10,128]
        P4_downsample = self.bu_conv1(P4_out)

        # [10,10,128] + [10,10,128] -> [10,10,256]
        P4_downsample = torch.cat([P4_downsample, P5], 1)

        P5_out = self.C3_n4(P4_downsample)

        # P3_out : [40,40,64]
        # P4_out : [20,20,128]
        # P5_out : [10,10,256]
        return (P3_out, P4_out, P5_out)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        # depthwise = True if phi == 'nano' else False
        depthwise = True
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs