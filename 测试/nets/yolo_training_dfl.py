#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pandas import np
from DFL.Integral import Integral
input_shape = [320,320]



class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="diou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type


    def forward(self, pred, target,eps=1e-7):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # cx,cy,w,h -> xmin,ymin,xmax,ymax
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == 'diou':
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            cw = c_br[..., 0] - c_tl[..., 0]
            ch = c_br[..., 1] - c_tl[..., 1]
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2 # (cx1-cx2)^2 + (cy1-cy2)^2
            diou = (iou - rho2 / c2).clamp(min=-1, max=1)
            loss =  1 - diou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class YOLOLoss(nn.Module):    
    def __init__(self, num_classes, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes        = num_classes
        self.strides            = strides

        self.bcewithlog_loss    = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss           = IOUloss(reduction="none")
        self.grids              = [torch.zeros(1)] * len(strides)

        self.Integral = Integral()

    def forward(self, inputs, labels=None):
        outputs             = []
        x_shifts            = []
        y_shifts            = []
        expanded_strides    = []
        regs_prob            = []
        #-----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        #-----------------------------------------------#


        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            # 共在三个不同大小的特征图上进行目标检测
            output, grid,reg_prob = self.get_output_and_grid(output, k, stride)
            # grid:[1,h*w,2]
            # 从grid中获得每个网格点的坐标分量x,y
            x_shifts.append(grid[:, :, 0])   # list: [[w1*h1],[w2*h2],[w3*h3]]
            y_shifts.append(grid[:, :, 1])   # list: [[w1*h1],[w2*h2],[w3*h3]]
            # expanded_strides:每个像素点的stride
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride) # list:[[w1*h1],[w2*h2],[w3*h3]]
            outputs.append(output) # outpus list:[[b,w1*h1,6],[b,w2*h2,6],[b,w3*h3,6]]
            regs_prob.append(reg_prob)  # list: [[b,w1*h1,68],[b,w2*h2,68],[b,w3*h3,68]]


        # torch.cat(outputs, 1),torch.cat(regs_prob, 1),此处将列表中三个张量在维度1进行拼接，因此 结果为:[b,w1*h1+w2*h2+w3*h3,6] 和 [b,w1*h1+w2*h2+w3*h3,68]
        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1),torch.cat(regs_prob, 1))


    def get_output_and_grid(self, output, k, stride,reg_max=14):
        """
        输入:
        1、output shape:[b,c,w,h]
        2、k: 表示输出来自第k个特征图
        3、stride: 第k个特征图对应的下采样倍数
        输出：
        1、output：解码后的output，shape:[b,w*h,c]
        2、grid:第k个特征图上的网格点的坐标,shape:[1,h*w,2]
        3、reg_out_pre_jifen:网络输出output的bbox regression部分在积分前的概率分布向量 shape:[b,w*h,4 * (reg_max+1)]  reg_max 默认为16
        """
        grid            = self.grids[k]
        hsize, wsize    = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            # yv, xv          = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # 输入[torch.arange(hsize), torch.arange(wsize)]: [行的索引范围,列的索引范围]
            # 输出yv, xv: 其中 yv为每一行所有网格点的行索引,shape为[hsize,wsize] ; xv为每一行所有网格点的列索引,shape为[hsize，wsize]
            yv, xv          = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])

            # torch.stack((xv, yv), 2)在第二维拼接: [hsize,wsize,2] -> view(1, hsize, wsize, 2) : [1,hsize,wsize,2]
            grid            = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())

            self.grids[k]   = grid
        # grid:[1,hsize,wsize,2] -> [1,hsize*wsize,2] 即对应每个锚点的坐标(y,x)
        grid                = grid.view(1, -1, 2)
        # output.shape = [b,c,w,h] -> flatten(start_dim=2): [b,c,w*h] -> permute(0, 2, 1): [b,w*h,c]
        output              = output.flatten(start_dim=2).permute(0, 2, 1)


        # 开始解码
        # 首先对网络输出的坐标回归的概率分布进行softmax和积分操作,获得锚点到预测框四条边的距离:l,t,r,b
        # 但这里的l,t,r,b是除以stride的结果，因此要乘stride变回去
        reg_output, reg_out_pre_jifen = self.Integral(output[..., :(reg_max+1)*4])
        output = torch.cat([reg_output, output[..., (reg_max+1)*4:(reg_max+1)*4+1], output[..., (reg_max+1)*4+1:]], 2)
        output[...,:4]  = output[...,:4] * stride

        # 将网格点映射回原图，获得原图上的预测框，然后将 l,t,r,b坐标转换为 xmin,ymin,xmax,ymax坐标
        output[...,:2] = grid * stride + 0.5 * stride - output[...,:2]
        output[...,2:4] = grid * stride + 0.5 * stride + output[...,2:4]
        # 由于网络初期，预测结果偏差比较大,因此加以约束,使坐标不会超出图片范围
        output[..., 0] = torch.clamp(output[..., 0].clone(), min=0)
        output[..., 1] = torch.clamp(output[..., 1].clone(), min = 0)
        output[..., 2] = torch.clamp(output[..., 2].clone(), max = wsize * stride-1)
        output[..., 3] = torch.clamp(output[..., 3].clone(), max = hsize * stride-1 )

        # 坐标变换 xmin,ymin,xmax,ymax -> cx,cy,w,h
        # cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
        # w = (xmax - xmin) / 2; h = (ymax - ymin) / 2
        output[...,0:2] =  (output[..., 0:2] + output[..., 2:4]) / 2
        output[..., 2:4] = output[..., 2:4] - output[..., 0:2]

        return output, grid,reg_out_pre_jifen



    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs,regs_prob,reg_max=14):
        """
        输入:
        1、x_shifts, y_shifts,expanded_strides 三张特征图上所有网格点的x,y坐标以及对应的stride ，list: [[w1*h1],[w2*h2],[w3*h3]]
        2、labels: 一个批次中所有图片中的ground_truth标签  list
        3、outputs: [b,w1*h1+w2*h2+w3*h3,6]
        4、regs_prob: [b,w1*h1+w2*h2+w3*h3,68]
        w1*h1+w2*h2+w3*h3 = all_anchor_points
        输出:
        该批次所有图片的reg loss,conf loss和分类loss之和除以所有正样本数
        """

        bbox_preds  = outputs[:, :, :4]   # [b,all_anchor_points,4]
        obj_preds   = outputs[:, :, 4:5]  # [b,all_anchor_points,1]
        cls_preds   = outputs[:, :, 5:]   # [b,all_anchor_points,num_classes]

        total_num_anchors   = outputs.shape[1]  # all_anchor_points
        #-----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        #-----------------------------------------------#
        # 每一个锚点对应的x坐标，y坐标,stride
        # list中所有tensor在维度1拼接 -> [1,all_anchor_points]
        x_shifts            = torch.cat(x_shifts, 1)
        y_shifts            = torch.cat(y_shifts, 1)
        expanded_strides    = torch.cat(expanded_strides, 1)
        grid = torch.stack((x_shifts,y_shifts),0).reshape(-1,2)  # [all_anchor_points,2]


        # 接下来进行label assignment
        # 首先声明四个list，分别存储正样本对应的gt的类别标签,回归标签,置信度标签以及正样本掩膜
        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks    = []
        # 记录正样本数量
        num_fg = 0.0

        # 再声明三个list，用于计算DFL_loss
        DFL_targets_x_shifts = []
        DFL_targets_y_shifts = []
        DFL_targets_strides = []

        gt_matched_classes_list = []
        for batch_idx in range(outputs.shape[0]):
            num_gt          = len(labels[batch_idx]) # 每幅图的gt个数
            if num_gt == 0:
                # 如果该图中没有gt的话，就初始化为0
                cls_target  = outputs.new_zeros((0, self.num_classes))
                reg_target  = outputs.new_zeros((0, 4))
                obj_target  = outputs.new_zeros((total_num_anchors, 1))
                fg_mask     = outputs.new_zeros(total_num_anchors).bool()
                #

                x_shifts_positive = x_shifts[0,fg_mask]
                y_shifts_positive = y_shifts[0,fg_mask]
                expanded_strides_positive = expanded_strides[0,fg_mask]
            else:
                #-----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, 4]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                #-----------------------------------------------#
                # batch_idx,一个batch中的第idx张
                # gt_bboxes_per_image：每张图片中gt的bbox坐标
                gt_bboxes_per_image     = labels[batch_idx][..., :4] # [num_gt,4]
                # gt_classes：每个gt的类别,要转为one_hot类型
                gt_classes              = labels[batch_idx][..., 4]

                # 每张图片中的预测框的坐标， 因为outputs的shape为[b, all_pred_bboxes, [cx,cy,w,h,conf,classes]]
                # 因此bbox_preds为[b,all_pred_bboxes,[cx,cy,w,h]], bboxes_preds_per_image为每张图片的预测框的坐标：[all_pred_bboxes,[cx,cy,w,h]]
                bboxes_preds_per_image  = bbox_preds[batch_idx]

                # 每张图片的[al_pred_bboxes,classes]
                cls_preds_per_image     = cls_preds[batch_idx]

                # 每张图片的[all_pred_bboxes,conf]
                obj_preds_per_image     = obj_preds[batch_idx]

                # 进行标签分配
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments( 
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts, 
                )
                gt_matched_classes_list.append(F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes))
                # gt_matched_classes：每个正样本对应的类别标签
                # fg_mask：正样本mask [2100]:其中正样本对应位置为True，反之为False
                # pred_ious_this_matching:正样本对应gt的IOU
                # matched_gt_inds：每个正样本匹配的gt
                # num_fg_img：每张图片的正样本总数

                num_fg      += num_fg_img
                cls_target  = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                # print(cls_target.shape)
                obj_target  = fg_mask.unsqueeze(-1)
                reg_target  = gt_bboxes_per_image[matched_gt_inds]  # [num_fg_img,4]
                ##############################################################################################################################################

                #   tensor转list: list = tensor.numpy().tolist()

                x_shifts_positive = x_shifts[0,fg_mask]   # 正样本的x坐标  [num_fg]
                y_shifts_positive = y_shifts[0,fg_mask]   # 正样本的y坐标  [num_fg]
                # print(x_shifts_positive.shape)
                expanded_strides_positive = expanded_strides[0,fg_mask]  # 正样本对应的下采样倍数 [num_fg]

                # torch.cuda.empty_cache()
                # cx,cy,w,h -> [l,t,b,r]






            ##################################################################################################

            DFL_targets_x_shifts.append(x_shifts_positive)  # [[num_fg1],[num_fg2],[num_fg3],[num_fg4]]
            DFL_targets_y_shifts.append(y_shifts_positive)  # [[num_fg1],[num_fg2],[num_fg3],[num_fg4]]
            DFL_targets_strides.append(expanded_strides_positive)  # [[num_fg1],[num_fg2],[num_fg3],[num_fg4]]
            #################################################################################################

            cls_targets.append(cls_target)  # [[num_fg1,classes_num],[num_fg2,classes_num],[num_fg3,classes_num],[num_fg4,classes_num]]
            reg_targets.append(reg_target)

            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        # 将list中的张量在维度0进行拼接 [[num_fg1,classes_num],[num_fg2,classes_num],[num_fg3,classes_num],[num_fg4,classes_num]] -> [batch_all_fg,num_classes]
        cls_targets = torch.cat(cls_targets, 0)  # 一个batch中所有图片的类别one-hot标签

        reg_targets = torch.cat(reg_targets, 0)  # torch.size([batch_all_fg,4]) 一个batch中所有图片的坐标回归目标: cx,cy,w,h

        obj_targets = torch.cat(obj_targets, 0)  # torch.size([batch_all_fg,1]) 一个batch中所有图片的置信度标签
        fg_masks    = torch.cat(fg_masks, 0)   # 一个batch中所有图片的fg_mask :torch.size([n_anchors_all * batch_size])

        gt_matched_classes_list = torch.cat(gt_matched_classes_list,0)
        # print(gt_matched_classes_list)

        # ###################### gt：cx,cy,w,h -> l,t,b,r
        # # DFL_targets_x_shifts：每个正样本点对应的特征图上的x坐标
        # # DFL_targets_y_shifts：每个正样本点对应的在特征图上的y左边
        # # DFL_targets_strides：每个正样本点对应的下采样倍数
        DFL_targets_x_shifts = torch.cat(DFL_targets_x_shifts,0).unsqueeze(1) # torch.size([batch_all_fg,1]]
        DFL_targets_strides = torch.cat(DFL_targets_strides,0).unsqueeze(1)  # torch.size([batch_all_fg,1]]
        DFL_targets_y_shifts = torch.cat(DFL_targets_y_shifts,0).unsqueeze(1)  # torch.size([batch_all_fg,1]]
        # # # 先将网格点映射回原图
        DFL_targets_x_shifts =  DFL_targets_x_shifts * DFL_targets_strides + 0.5 * DFL_targets_strides  # 正样本点在原图上的x坐标  torch.size([batch_all_fg,1]]

        DFL_targets_y_shifts = DFL_targets_y_shifts * DFL_targets_strides + 0.5 * DFL_targets_strides  # 正样本点在原图上的y坐标  torch.size([batch_all_fg,1]]
        # # # 然后将gt的cx,cy,w,h转化为 l,t,r,b形式
        DFL_targets = torch.zeros_like(reg_targets)  # torch.size([batch_all_fg,4]]

        # 坐标变换，将reg_targets的[cx,cy,w,h]形式的坐标变成[l,t,r,b]形式的坐标
        # 第一步: cx,cy,w,h -> xmin,ymin,xmax,ymax
        DFL_targets[...,0:2] = reg_targets[...,0:2] - 0.5 * reg_targets[...,2:4]
        DFL_targets[...,2:4] = reg_targets[...,0:2] + 0.5 * reg_targets[...,2:4]
        # 第二步: xmin,ymin,xmax,ymax -> l,t,r,b
        DFL_targets[...,0] = DFL_targets_x_shifts[...,0] - DFL_targets[...,0]  # l = anchor_x - xmin
        DFL_targets[...,1] = DFL_targets_y_shifts[...,0] - DFL_targets[...,1]  # t = anchor_y - ymin
        DFL_targets[...,2] = DFL_targets[...,2] - DFL_targets_x_shifts[...,0]  # r = xmax - anchor_x
        DFL_targets[...,3] = DFL_targets[..., 3] - DFL_targets_y_shifts[...,0] # b = ymax - anchor_y
        # 第三步: l,t,r,b / strides 映射到[0,reg_max]区间内
        DFL_targets = DFL_targets / DFL_targets_strides

        # 离label左边最近的值
        DFL_targets_left = DFL_targets.to(torch.long)  # torch.size([batch_all_fg,4]]
        DFL_targets_left[..., 0:4] = torch.clamp(DFL_targets_left[..., 0:4].clone(), min=0, max=reg_max)
        # 离label右边最近的值
        DFL_targets_right = (DFL_targets_left + 1) # torch.size([batch_all_fg,4]]
        DFL_targets_right[..., 0:4] = torch.clamp(DFL_targets_right[..., 0:4].clone(),min=0,max=reg_max)
        # regs_prob [b, w1 * h1 + w2 * h2 + w3 * h3, 68]
        DFL_pred_target = regs_prob.reshape(-1,4,reg_max+1)[fg_masks]  # [batch_all_fg,4,17]


        # DFL LOSS = (y_right - y) * log(p_left) + (y - y_left) * log(p_right)
        weight_left = DFL_targets_right.to(torch.float) - DFL_targets   # y_right - y   torch.size([batch_all_fg,4]]
        weight_left[...,0:4] = torch.clamp(weight_left[..., 0:4].clone(), min=0)  #


        num_fg = max(num_fg, 1)
        weight_right = DFL_targets - DFL_targets_left.to(torch.float)  # y - y_left
        weight_right[..., 0:4] = torch.clamp(weight_right[..., 0:4].clone(), min=0)

        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        print(cls_preds.shape)
        print(cls_targets.shape)
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()

        ###############################################################################################################


        loss_left = F.cross_entropy(
            DFL_pred_target.view(-1, reg_max+1), DFL_targets_left.view(-1), reduction='none').view(
            DFL_targets_left.shape) * weight_left

        loss_right = F.cross_entropy(
            DFL_pred_target.view(-1, reg_max+1), DFL_targets_right.view(-1), reduction='none').view(
            DFL_targets_right.shape) * weight_right

        loss_dfl = (loss_left + loss_right).mean(-1, keepdim=True).sum()





        reg_weight  = 5.5
        loss = reg_weight * loss_iou + loss_obj + loss_cls + 0.25 * loss_dfl
        return loss / num_fg

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        # 参数介绍：
        # num_gt：每张图片中的gt总数
        # total_num_anchors：所有级别特征图中的锚点总数，也是所有级别特征图中的像素点总数
        # gt_bboxes_per_image：每张图中的gt_bbox坐标
        # gt_classes:每张图中的gt的类别
        # bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image：每张图中预测框输出的bbox坐标，类别向量和confidence
        # expanded_strides:每一个锚点对应的stride
        # x_shifts, y_shifts： 每一个锚点对应的x,y
        # 假设输入图片大小为[320,320]，则expanded_strides.shape = [1,2100], 而x_shifts.shape, y_shifts.shape = [1,2100]

        #-------------------------------------------------------#
        #   fg_mask                 [total_num_anchors]
        #   is_in_boxes_and_center  [num_gt, 候选正样本数]
        #-------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)

        #-------------------------------------------------------#
        #   fg_mask                 [total_num_anchors]
        #   bboxes_preds_per_image  [候选正样本数, 4]
        #   cls_preds_              [候选正样本数, num_classes]
        #   obj_preds_              [候选正样本数, 1]
        #-------------------------------------------------------#

        # 取出所有预测框中的候选正样本对应的预测框的坐标，conf以及类别
        bboxes_preds_per_image  = bboxes_preds_per_image[fg_mask]
        cls_preds_              = cls_preds_per_image[fg_mask]
        obj_preds_              = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor     = bboxes_preds_per_image.shape[0] # 候选正样本数

        #-------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        #-------------------------------------------------------#
        # 计算每个gt和所有候选正样本的iou
        pair_wise_ious      = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False) # [num_gt,候选正样本数]

        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        #-------------------------------------------------------#
        #   cls_preds_          [num_gt, 候选正样本数, num_classes]
        #   gt_cls_per_image    [num_gt, 候选正样本数, num_classes]
        #-------------------------------------------------------#
        # cls_preds_:[候选正样本数,num_classes] -> 在第0维加一维：unsqueeze(0)，得到[1,候选正样本数,num_classes] -> 第0维重复num_gt次 repeat(num_gt, 1, 1) 得到[num_gt,候选正样本数,num_classes]
        cls_preds_          = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        # 将类别标签转化为one-hot,然后在第一维加一个维度，并重复num_in_boxes_anchor次
        gt_cls_per_image    = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1) # [num_gt,候选正样本数,num_classes]
        # 计算交叉熵损失
        pair_wise_cls_loss  = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_
        # 计算cost矩阵，对非候选样本点置为很大的数值，cost越小越可能成为真正的正样本点
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg
    
    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1) # bboxes_a的面积：w*h
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1) # bboxes_b的面积
        else:
            tl = torch.max(              # [num_gt,num_yuce,2]
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )

            br = torch.min(              # [num_gt,num_yuce,2]
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)  # shape: [num_gt]
            area_b = torch.prod(bboxes_b[:, 2:], 1)  # shape: [num_yuce]

        en = (tl < br).type(tl.type()).prod(dim=2)  # shape: [num_gt,num_yuce]

        area_i = torch.prod(br - tl, 2) * en   # [num_gt,num_yuce]

        return area_i / (area_a[:, None] + area_b - area_i) # [num_gt,num_yuce]

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt, center_radius = 2.5):
        """
        粗标签分配：
        1、首先计算锚点是否落入gt范围内：
        (1)将网格点映射回原图：x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)，
        同时因为所有锚点都要判断是否落入某个gt中，因此将这个网格点映射回原图的坐标矩阵repeat num_gt次
        (2)为了便于计算锚点到gt四条边的距离，先将gt的(cx,cy,w,h)转换为(xmin,ymin,xmax,ymax)的形式。
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        最终每个tensor的shape为:[num_gt,2100]
        (3)计算锚点到gt四条边的距离：
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        这四个tensor中每一行表示2100个锚点离某个gt四条边的距离
        最后将这四个tensor堆叠在一起：bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  shape:[num_gt,2100,4]
        (4)判断锚点是否落入某个gt:
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        然后将所有结果综合一下,即获得所有的候选正样本点：
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  shape:[2100]  里面的值为True Or False 来表示某个锚点是否是候选正样本点
        2、计算锚点是否落入以gt中心外扩边长为2*2.5*stride的矩形的范围内,这里和前面的操作基本一致，只是在第一步有所不同：
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        3、最后将1，2步的结果综合一下：
        is_in_boxes_anchor      = is_in_boxes_all | is_in_centers_all   shape:[2100]
        is_in_boxes_and_center  = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]  shape:[num_gt,2100]

        """
        #-------------------------------------------------------#
        #   expanded_strides  [1,total_num_anchors]
        #   x_shifts          [1,total_num_anchors]
        #   y_shifts          [1,total_num_anchors]
        #   x_centers_per_image  [num_gt,total_num_anchors]
        #   y_centers_per_image  [num_gt,total_num_anchors]
        #-------------------------------------------------------#
        expanded_strides_per_image  = expanded_strides[0]  # expanded_strides:[1,total_num_anchors],   expanded_strides_per_image :[total_num_anchors]
        # x_centers_per_image:特征点映射回原图的坐标
        # (x_shifts[0] + 0.5) * expanded_strides_per_image : [total_num_anchors]
        # ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0): 在第0维加一个维度-> [1,total_num_anchors]
        # .repeat(num_gt, 1): 函数repeat，但参数只有两个时，表示列重复次数和行重复次数，参数有三个时，表示通道重复次数，列重复次数，行重复次数。而重复次数为1表示不重复。
        # 所以.repeat(num_gt, 1)表示列重复num_gt次，行不重复
        # x_centers_per_image.shape : [num_gt,total_num_anchors]
        x_centers_per_image         = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        # y_centers_per_image : [num_gt,2100]
        y_centers_per_image         = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        #-------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        #-------------------------------------------------------#
        # cx,cy,w,h -> xmin,ymin,xmax,ymax
        # gt_bboxes_per_image.shape = [num_gt,4]
        # gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]) : [num_gt]
        # unsqueeze(1): [num_gt,1]
        # .repeat(1, total_num_anchors) : [num_gt,total_num_anchors]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)

        #-------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        #   cx - xmin = l
        #   xmax - cx = r
        #   cy - ymin = t
        #   ymax - cy = b
        #-------------------------------------------------------#
        # 计算每个锚点到gt四条边的距离：
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # bbox_deltas: [num_gt,total_num_anchors,4]
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        #-------------------------------------------------------#
        #   is_in_boxes     [num_gt, total_num_anchors]
        #   is_in_boxes_all [total_num_anchors]
        #-------------------------------------------------------#
        # 一个锚点是否落入某个gt中的判断方法是锚点离gt四条边的距离都大于0
        # bbox_deltas.min(dim=-1).values > 0.0: 首先按最后一维取最小值，然后将最小值大于0的置为True，反之为False
        is_in_boxes     = bbox_deltas.min(dim=-1).values > 0.0  # [num_gt, total_num_anchors]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # [total_num_anchors]

        # 以gt为中心往外扩展center_radius * expanded_strides_per_image，得到一个大的矩形范围：[xmin,ymin,xmax,ymax]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

        #-------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        #-------------------------------------------------------#

        # 计算锚点外扩后的gt四条边的距离
        # 当c_l>0,c_r>0,c_t>0,c_b>0，那么这个锚点就落入判定矩形中，被认为是候选的正样本
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas       = torch.stack([c_l, c_t, c_r, c_b], 2)

        #-------------------------------------------------------#
        #   is_in_centers       [num_gt, total_num_anchors]
        #   is_in_centers_all   [total_num_anchors]
        #-------------------------------------------------------#
        is_in_centers       = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all   = is_in_centers.sum(dim=0) > 0

        #-------------------------------------------------------#
        #   is_in_boxes_anchor      [total_num_anchors]  它可以看作是候选正样本掩膜
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]   矩阵：行数为num_gt
        #-------------------------------------------------------#
        is_in_boxes_anchor      = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center  = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):

        #-------------------------------------------------------#
        #   输入:
        #   fg_mask:候选正样本点
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]        
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        #-------------------------------------------------------#
        matching_matrix         = torch.zeros_like(cost)

        #------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        #------------------------------------------------------------#
        n_candidate_k           = min(10, pair_wise_ious.size(1)) # n_candidate_k：如果候选的正样本数小于10，就为候选正样本数；否则为10
        topk_ious, _            = torch.topk(pair_wise_ious, n_candidate_k, dim=1)  # 为每个gt找n_candidate_k个IOU最大的点  [num_gt,n_candidate_k]
        dynamic_ks              = torch.clamp(topk_ious.sum(1).int(), min=1) # 然后把这10个IOU求和，并确保IOU最小为1 [num_gt,1] 每一行表示要为这个gt分配的正样本个数

        for gt_idx in range(num_gt):
            #------------------------------------------------------------#
            #   给每个真实框选取cost最小的动态k个点
            #------------------------------------------------------------#
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False) # cost: [num_gt, fg_mask]  从fg_mask个候选样本中挑出k个cost最小的点
            matching_matrix[gt_idx][pos_idx] = 1.0 # 标记挑选出来的点 [num_gt,fg_mask]
        del topk_ious, dynamic_ks, pos_idx

        #------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        #------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0)  # [fg_mask] 一个锚点可能对应多个gt，求和后如果大于1，那么说明这个锚点与多个gt匹配，要做处理
        # 如果
        if (anchor_matching_gt > 1).sum() > 0:
            #------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            #------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0) # 取出cost最小的gt
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0 # 先将所有大于1的位置的值置为0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0 # 然后将cost最小的行中sum大于1的点的值置为1
        #------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        #------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0  # 最后获得真正的正样本点的mask shape:[候选正样本总数]

        num_fg          = fg_mask_inboxes.sum().item()  # 正样本总数


        #------------------------------------------------------------#
        #   对fg_mask进行更新,[2100]中正样本对应位置为true，其他为false
        #------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes


        #------------------------------------------------------------#
        #   获得特征点对应的物品种类
        #------------------------------------------------------------#
        # 每一列找值最大的那一行(值为1的行),并存下该行的索引
        # 例如结果为：[1, 3, 3, 3, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 4, 4, 4]，表示有18个正样本，数字表示每一个正样本与哪个gt相匹配
        matched_gt_inds     = matching_matrix[:, fg_mask_inboxes].argmax(0)
        # 获得每个正样本点对应gt的类别
        gt_matched_classes  = gt_classes[matched_gt_inds]

        # 正样本与其对应的gt的IOU
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

