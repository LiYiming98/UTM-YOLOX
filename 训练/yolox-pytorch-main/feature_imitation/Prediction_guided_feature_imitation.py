import math

import torch

import torch.nn.functional as F
import torch.nn as nn
# from Attention.Cbam import SpatialAttention
import torch

class Prediction_guided_feature_imitation_loss(nn.Module):

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 ):
        super(Prediction_guided_feature_imitation_loss, self).__init__()

        self.align = nn.Conv2d(student_channels,teacher_channels,kernel_size=1,stride=1, padding=0)


    def forward(self,
                tea_cls_pre,
                stu_cls_pre,
                T_fpn_output,
                S_fpn_output):
        """Forward function.
        param:
        1、tea_cls_pre :  [b,w*h,2]
        2、stu_cls_pres : [b,w*h,2]
        3、T_fpn_output : [b,c,h,w]
        4、S_fpn_output : [b,c_s,h,w]

        method :
        1、 P_dif = sum((stu_cls_pre - tea_cls_pre) ** 2) / C : C 通道数
        2、 F_dif = l2_loss(stu_F - tea_F) / Q  : Q 通道数
        3、 LOSS_PFI = (P_dif * F_dif) ^ 2  / (H * W)  / L : l:使用的特征图的数量
        """
        b,c,h,w = T_fpn_output.size()
        device = T_fpn_output.device

        P_dif = ((((tea_cls_pre - stu_cls_pre) ** 2).sum(dim = 2) / 2).reshape(b,h,w)).to(device) # b,h,w

        S_fpn_output_align = (self.align(S_fpn_output)).to(device) # b,c,h,w
        F_dif = (((T_fpn_output - S_fpn_output_align) ** 2).sum(dim=1) / c).to(device) # b,h,w

        loss = (self.get_dis_loss(P_dif,F_dif)).to(device)

        return loss

    def get_dis_loss(self,P_dif,F_dif):
        b,h,w = P_dif.size()
        loss_mse = nn.MSELoss(reduction='mean')
        loss = loss_mse(P_dif.mul(F_dif),torch.zeros([b,h,w]).cuda())
        return loss



