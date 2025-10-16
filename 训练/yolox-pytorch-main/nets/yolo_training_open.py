#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import math
from copy import deepcopy
from functools import partial
from nets.ICloss import memory_bank
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
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

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOLoss(nn.Module):
    def __init__(self, num_classes, fp16, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(strides)
        self.fp16 = fp16
        self.memory_bank = memory_bank(5, 0.1, 200)

    def forward(self, inputs,IC_outputs, oln_outputs, unknown_outputs ,epoch, labels=None,is_training=True):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # -----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        # -----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), IC_outputs, oln_outputs, unknown_outputs,epoch,is_training)

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        grid = grid.view(1, -1, 2)

        output = output.flatten(start_dim=2).permute(0, 2, 1)
        output[..., :2] = (output[..., :2] + grid.type_as(output)) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid
    
    def get_IC_Loss(self,feats,ious,gt_cls_all,oln_preds2,epoch):
#         return self.memory_bank.get_IC_loss(feats,gt_cls_all,ious,epoch)
        return self.memory_bank.Prototype_IC_loss(feats,gt_cls_all,ious,oln_preds2,epoch)

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs, IC_outputs, oln_outputs, unknown_outputs,epoch,is_training=True):
        # -----------------------------------------------#
        #   [batch, n_anchors_all, 4]
        # -----------------------------------------------#
        bbox_preds = outputs[:, :, :4]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, 1]
        # -----------------------------------------------#
        obj_preds = outputs[:, :, 4:5]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, n_cls]
        # -----------------------------------------------#
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        # -----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        # -----------------------------------------------#
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        # IC loss 计算中的 ious
        ic_needed_ious = []
        ic_needed_cls = []
        ic_needed_feats = []
        IC_outputs = torch.cat(IC_outputs, dim=1)  # [b,2100,128]
        oln_outputs = torch.cat(oln_outputs, dim=1)  # [b,all_anchors,1]
        unknown_outputs = torch.cat(unknown_outputs, dim=1)
        cls_targets_one_hot = []
        oln_preds = []
        oln_targets = []
        unknown_fg_masks = []
        unknown_bg_masks = []
        
        oln_preds_all = oln_outputs.view(-1,1)


        
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                unknown_fg_mask = outputs.new_zeros(total_num_anchors).bool()
                unknown_bg_mask = outputs.new_ones(total_num_anchors).bool()
                
            else:
                # -----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, num_classes]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                # -----------------------------------------------#
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts,
                )
                IC_output = IC_outputs[batch_idx]  # [2100,128]


                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # cls_target_one_hot = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes)  #


                cls_target = F.one_hot(gt_matched_classes.to(torch.int64),
                                       self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
#                 cls_target[cls_target != 0] = cls_target[cls_target != 0]
#                 cls_target[cls_target == 0] = 0.1
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                # OLN target: 计算 预测框和gt的iou
                iou_all = self.bboxes_iou(bboxes_preds_per_image, gt_bboxes_per_image, False)
                iou_all_target, iou_max_idx = torch.max(iou_all, dim=1)
                
                unknown_fg_mask = (iou_all_target >= 0.7).bool() # 如果预测框和真实框的IoU 大于等于0.5,则视为前景
                unknown_bg_mask = (iou_all_target <= 0.3).bool()
                oln_targets.append(iou_all_target)  # 每一个预测框哪个gt的最大的iou
                oln_output = oln_outputs[batch_idx]
                oln_preds.append(oln_output)

                # ic taregt
                iou, ic_gt_cls = self.get_ic_assignments(num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                                                         bboxes_preds_per_image, cls_preds_per_image)
                ic_needed_ious.append(iou)
                ic_needed_cls.append(ic_gt_cls)
                ic_needed_feats.append(IC_output)



            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)
            unknown_fg_masks.append(unknown_fg_mask)
            unknown_bg_masks.append(unknown_bg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        unknown_fg_masks = torch.cat(unknown_fg_masks,0)
        unknown_bg_masks = torch.cat(unknown_bg_masks,0)

        # IC LOSS
        """
        ious : torch.Size([pos_nums])
        gt_cls_all : torch.Size([pos_nums, 1])
        feats : torch.Size([pos_nums, 128])
        """
        loss_ic = obj_preds.new_tensor(0.0)

        if num_fg != 0 and epoch >= 0 and is_training:
            oln_preds2 = torch.cat(oln_preds, 0)
            ic_needed_ious = torch.cat(ic_needed_ious,0) # 一个Batch内 初筛出的所有 pred boxes 和 gt的最大IOU
       
            ic_needed_cls = torch.cat(ic_needed_cls,0).unsqueeze(1) # 一个Batch内 初筛出的所有pred boxes 对应的类别标签
         
            ic_needed_feats = torch.cat(ic_needed_feats,0)
           
            loss_ic = ic_needed_feats.new_tensor(0.0)
            loss_ic = self.get_IC_Loss(ic_needed_feats,ic_needed_ious,ic_needed_cls,oln_preds2,epoch)*0.5
        
        ## oln loss
        
        
        loss_oln = cls_preds.new_tensor(0.0)
        
        ########################################
        if len(oln_targets) != 0:
            oln_targets = torch.cat(oln_targets, 0)
            oln_preds = torch.cat(oln_preds, 0)
            
#             oln_preds = torch.sigmoid(oln_preds)
            #######################################
            idx = oln_targets >= 0.3
            OLN_target = oln_targets[idx].unsqueeze(dim=1)
            OLN_pred = oln_preds[idx, :]

            N, _ = OLN_target.shape

            S = (int)(N / 2)
       
            if S != 0:
                index = torch.LongTensor(random.sample(range(N), S)).to(OLN_pred.device)

                OLN_target = torch.index_select(OLN_target, 0, index).to(OLN_pred.device)
                OLN_pred = torch.index_select(OLN_pred, 0, index)

                loss_oln = F.l1_loss(OLN_pred, OLN_target, reduction='mean')
#             smoothloss= nn.SmoothL1Loss(reduction='mean')
#             loss_oln = smoothloss(OLN_pred,OLN_target)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        
#         alpha = 0.25
#         gamma = 2.0
#         alpha_factor = torch.ones_like(cls_targets) * alpha
         
#         pred = torch.sigmoid(cls_preds.view(-1, self.num_classes)[fg_masks])
#         alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
#         focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - pred, pred)
#         focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
        
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets*0.9)).sum()
        
        
#         x = torch.sigmoid(cls_preds.view(-1, self.num_classes)[fg_masks])
#         x = torch.clamp(x, min=1e-7, max=1-1e-7)
#         loss_cls = - cls_targets * torch.log(x) - (1-cls_targets) * torch.log(1-x)
        
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls
        
    
        
        if epoch >=85:
            unknown_outputs = unknown_outputs.view(-1,1)
            cls_preds = cls_preds.view(-1, self.num_classes)
            loss_up = self.get_up_loss2(cls_preds[unknown_fg_masks],
                                       unknown_outputs[unknown_fg_masks],oln_preds_all.squeeze()[unknown_fg_masks])
            
#             loss_up = self.get_up_loss(cls_targets, cls_preds[fg_masks],
#                                        unknown_outputs[fg_masks],oln_preds_all.squeeze()[fg_masks])
            
            
            loss_up = loss_up if not torch.isnan(loss_up) else cls_preds.new_tensor(0.0)
            
            
#             bg_masks = ~ fg_masks
            
#             loss_up_bg = self.UP_loss_bg5(cls_preds.view(-1, self.num_classes),unknown_outputs.view(-1,1),oln_preds_all.squeeze(),obj_preds.view(-1, 1),bg_masks)
            
            loss_up_bg = self.UP_loss_bg6(cls_preds.view(-1, self.num_classes),unknown_outputs.view(-1,1),oln_preds_all.squeeze(),obj_preds.view(-1, 1),unknown_bg_masks)
            loss_up_bg = loss_up_bg if not torch.isnan(loss_up_bg) else cls_preds.new_tensor(0.0)
            
            loss = loss / num_fg  + loss_ic +  0.5*loss_up_bg + 0.5 * loss_up
            
        else:
            loss = loss / num_fg + loss_ic 
            
        
#         loss = loss if not torch.isnan(loss) else cls_preds.new_tensor(0.0)
        # 啥也不用
#         return loss
        return loss + loss_oln * 4
    
    def UP_loss_bg6(self,classification,unknown_pred,oln_out,obj_pred,negative_indices):
         # 负样本的类别概率
        neg_cls = classification[negative_indices, :]  # [neg_num,num_classes]
        neg_obj = obj_pred[negative_indices,:]
        
        # 负样本的未知类概率
        neg_unknown_pred = torch.sigmoid(unknown_pred[negative_indices, :])  # [pos_num,1]
        
        
        p = torch.sigmoid(neg_cls)
        p2 = torch.sigmoid(neg_cls)
        p = torch.cat([p,neg_unknown_pred],dim=1)
        oln_pred = oln_out[negative_indices] # [neg_num]
        idx = oln_pred >= 0.7  # 0.7
        val,_ = torch.max(p2,dim=1)
#         idx2 = val >= 0.2  # 0.6 
#         idx = torch.logical_and(idx.squeeze(), idx2)
        p = torch.sum(p[idx,:],dim=1)
        
        x = p.shape[0]
        
        if x > 10:
            x = 10
        _,idx5 = torch.topk(p,x)
        
        
        
        a = neg_unknown_pred[idx.unsqueeze(dim=1)].unsqueeze(dim=1)
        
        
        
        
        a = torch.cat([torch.sigmoid(neg_cls)[idx,:],a],dim=1)
#         a = a[idx5]
        
        a = torch.clamp(a, min=1e-7, max=1-1e-7)
        targets = torch.zeros_like(a)
        targets[:,-1] = 0.8
        
        
        loss_unknown = torch.tensor(0).type_as(unknown_pred)
        print(a.shape)
        if a.shape != torch.Size([0]):
            loss_unknown = (- targets * torch.log(a) - (1- targets) * torch.log(1-a)).sum() / a.shape[0]
        p2 = torch.sigmoid(neg_cls)
        val5,_ = torch.max(p2,dim=1)
        idx = oln_pred >= 0.7
        idx3 = val5 <= 0.1
        idx4 = torch.logical_and(idx.squeeze(), idx3)
        b = neg_unknown_pred[idx4.unsqueeze(dim=1)]
        b = torch.clamp(b,min=1e-7,max = 1-1e-7)
        targets2 = torch.ones_like(b) * 0
        loss_bg =  torch.tensor(0).type_as(unknown_pred)
        
        if b.shape != torch.Size([0]):
            loss_bg = -(1-targets2) * torch.log(1-b)
        
 
        loss = loss_unknown.mean() + loss_bg.mean()
       
        return loss

    def UP_loss_bg8(self,classification,unknown_pred,oln_out,obj_pred,negative_indices):
         # 负样本的类别概率
        neg_cls = classification[negative_indices, :]  # [neg_num,num_classes]
        neg_obj = obj_pred[negative_indices,:]
        
        # 负样本的未知类概率
        neg_unknown_pred = torch.sigmoid(unknown_pred[negative_indices, :])  # [pos_num,1]
        x = torch.sigmoid(neg_cls)
#         x = torch.cat([x,neg_unknown_pred],dim=1)
      
        p = torch.sigmoid(neg_obj)
        p = torch.cat([p,neg_unknown_pred],dim=1)
        
        oln_pred = oln_out[negative_indices] # [neg_num]
        idx = oln_pred >= 0.75 # 0.7
#         val11,_ = torch.max(torch.sigmoid(neg_cls),dim=1)
#         idx11 = val11 >= 0.5
#         idx = torch.logical_and(idx.squeeze(), idx11)
        
        
        x = x[idx,:]
        
        p = p[idx,:]
        
        x = torch.sum(x,dim=1)
        
        n = x.shape[0]
        if n >= 10:
            n = 10
            
        loss_unknown = torch.tensor(0).type_as(oln_pred)
        if x.shape != torch.Size([0]):
            
            val,idx2 = torch.topk(x,n)
            
            p = p[idx2,:]
            print(p.shape[0])
        
            targets = torch.zeros_like(p)
            targets[:,-1] = 0.73

            loss_unknown = (- targets * torch.log(p) - (1- targets) * torch.log(1-p)).sum() / p.shape[0] 
            
        idx = oln_pred >= 0.3
        val,_ = torch.max(torch.sigmoid(neg_cls),dim=1)
        idx3 = val <= 0.2 
        
        idx4 = torch.logical_and(idx.squeeze(), idx3)
        b = neg_unknown_pred[idx4.unsqueeze(dim=1)]
        b = torch.clamp(b,min=1e-7,max = 1-1e-7).unsqueeze(dim=1)
        
        b = torch.cat([torch.sigmoid(neg_obj[idx4,:]),b],dim=1)
        
        
        targets2 = torch.ones_like(b) * 0
        loss_bg =  torch.tensor(0).type_as(unknown_pred)
        # 仅背景采样
       
        if b.shape != torch.Size([0]):
            loss_bg = (-(1-targets2) * torch.log(1-b)).sum() / b.shape[0]
        
 
        loss = loss_unknown + loss_bg
       
        return loss
    
    def UP_loss_bg7(self,classification,unknown_pred,oln_out,obj_pred,negative_indices):
         # 负样本的类别概率
        neg_cls = classification[negative_indices, :]  # [neg_num,num_classes]
        neg_obj = obj_pred[negative_indices,:]
        
        # 负样本的未知类概率
        neg_unknown_pred = torch.sigmoid(unknown_pred[negative_indices, :])  # [pos_num,1]

        p = torch.cat([neg_unknown_pred,torch.sigmoid(neg_cls)],dim=1)
        oln_pred = oln_out[negative_indices] # [neg_num]
        idx = oln_pred >= 0.7 # 0.7
        val,_ = torch.max(p,dim=1)
        
        idx2 = val >= 0.55  # 0.6 
        idx = torch.logical_and(idx.squeeze(), idx2)
        
        
        a = neg_unknown_pred[idx.unsqueeze(dim=1)]
       
        a = torch.clamp(a, min=1e-7, max=1-1e-7).unsqueeze(dim=1)
        
        a = torch.cat([torch.sigmoid(neg_cls[idx,:]),a],dim=1)
        
        targets = torch.zeros_like(a)
        targets[:,-1] = 1
        
        loss_unknown = torch.tensor(0).type_as(unknown_pred)
        
#         print(a)
#         print(targets)
        print(a.shape)
        if a.shape != torch.Size([0]):
            loss_unknown = (- targets * torch.log(a) - (1- targets) * torch.log(1-a)).sum() / a.shape[0] 
            
        idx = oln_pred >= 0.4
        idx3 = val <= 0.2
        idx4 = torch.logical_and(idx.squeeze(), idx3)
        b = neg_unknown_pred[idx4.unsqueeze(dim=1)]
        b = torch.clamp(b,min=1e-7,max = 1-1e-7).unsqueeze(dim=1)
        
        b = torch.cat([torch.sigmoid(neg_cls[idx4,:]),b],dim=1)
        
        
        targets2 = torch.ones_like(b) * 0
        loss_bg =  torch.tensor(0).type_as(unknown_pred)
        # 仅背景采样
        if b.shape != torch.Size([0]):
            loss_bg = (-(1-targets2) * torch.log(1-b)).sum() / b.shape[0]
        
 
        loss = loss_unknown + loss_bg
       
        return loss
    
    
    def UP_loss_bg4(self,classification,unknown_pred,oln_out,negative_indices):
        # 负样本的类别概率
        neg_cls = classification[negative_indices, :]  # [neg_num,num_classes]
        # 负样本的未知类概率
        neg_unknown_pred = unknown_pred[negative_indices, :]  # [pos_num,1]


        oln_pred = oln_out[negative_indices] # [neg_num]
        
        idx = oln_pred >= 0.7
        val = torch.sum(torch.sigmoid(neg_cls),dim=1)
        
        idx2 = val >= 0.5
        idx = torch.logical_and(idx.squeeze(), idx2)
        
        a = neg_unknown_pred[idx.unsqueeze(dim=1)]
        
        
        a = torch.clamp(a, min=1e-7, max=1-1e-7)
        
        if a.shape == torch.Size([0]):
            return torch.tensor(0).type_as(unknown_pred)
        # 本来是0.7 改为 0.5
        
        shape = a.shape
        N = shape[0]
        S =(int)(N / 2)
            
#         index  = torch.LongTensor(random.sample(range(N), S)).to(oln_out.device)

#         a = torch.index_select(a, 0, index).to(oln_out.device)
#         OLN_pred = torch.index_select(OLN_pred, 0, index)

        confidence = oln_pred[idx]
        _,idx3 = torch.topk(confidence,S)
        
        a = torch.sigmoid(a[idx3]) 
        a = torch.clamp(a, min=1e-7, max=1-1e-7)
        
        targets = 0.7 * torch.ones_like(a)
        loss = - targets * torch.log(a) - (1-targets) * torch.log(1-a)
        
        return loss.mean()
    
    
    
    def UP_loss_bg3(self,classification,unknown_pred,oln_out,negative_indices):
        # 负样本的类别概率
        neg_cls = classification[negative_indices, :]  # [neg_num,num_classes]
        # 负样本的未知类概率
        neg_unknown_pred = torch.sigmoid(unknown_pred[negative_indices, :])  # [pos_num,1]


        oln_pred = torch.sigmoid(oln_out[negative_indices]) # [neg_num]
        
        idx = oln_pred >= 0.4
        val = torch.sum(torch.sigmoid(neg_cls),dim=1)
        
        idx2 = val >= 0.1
        idx = torch.logical_and(idx.squeeze(), idx2)
        
        
        
        a = neg_unknown_pred[idx.unsqueeze(dim=1)]
        
        
        a = torch.clamp(a, min=1e-7, max=1-1e-7)
        print(a.shape)
        if a.shape == torch.Size([0]):
            return torch.tensor(0).type_as(unknown_pred)
        # 本来是0.7 改为 0.5
        
        n = a.shape
        print(n)
        n = n.numel()
        if n > 10:
            n = 10
        
        confidence = neg_unknown_pred.squeeze()[idx] * oln_pred[idx]
        _,idx3 = torch.topk(confidence,n)
        
        a = a[idx3] 
       
        
        targets = 0.6 * torch.ones_like(a)
        loss = - targets * torch.log(a) - (1-targets) * torch.log(1-a)
       
        return loss.mean()
    
    def UP_loss_bg5(self,classification,unknown_pred,oln_out,obj_pred,negative_indices):
        # 负样本的类别概率
        neg_cls = classification[negative_indices, :]  # [neg_num,num_classes]
        neg_obj = obj_pred[negative_indices,:]
        
        # 负样本的未知类概率
        neg_unknown_pred = torch.sigmoid(unknown_pred[negative_indices, :])  # [pos_num,1]


        oln_pred = oln_out[negative_indices] # [neg_num]
        
        idx = oln_pred >= 0.7
        val,_ = torch.max(torch.sigmoid(neg_cls),dim=1)
        
        idx2 = val >= 0.6
        idx = torch.logical_and(idx.squeeze(), idx2)
        
        
        
        a = neg_unknown_pred[idx.unsqueeze(dim=1)]
        
        
        a = torch.clamp(a, min=1e-7, max=1-1e-7)
        
        if a.shape == torch.Size([0]):
            return torch.tensor(0).type_as(unknown_pred)
        # 本来是0.7 改为 0.5
        
#         n = a.shape
#         n = n.numel()
#         if n > 15:
#             n = 15
        
#         confidence = oln_pred[idx]
#         _,idx3 = torch.topk(confidence,n)
        
#         a = a[idx3] 
       
        
        targets = torch.ones_like(a) * 0.7
        loss = - targets * torch.log(a) - (1-targets) * torch.log(1-a)
#         loss = - targets * torch.log(a)
        return loss.mean()




    
    
    def get_up_loss(self, pos_target, pos_cls, pos_unknown_pred,iou):


        # 1. Max entropy
        pos_entropy = (- torch.mul(pos_cls, torch.log(pos_cls))).sum(1)  # [pos_num] 每一个正样本的交叉熵
        sorted, max_entropy_indices = torch.sort(pos_entropy, descending=True)  # 降序排序
        max_entropy_indices = max_entropy_indices[:30]  # [10]

        hard_example_indice = max_entropy_indices

        scores = pos_cls[hard_example_indice]  # hard_num,num_classes
        # 正样本 最大类别概率和对应的标签

        # 如果没有正样本
        if pos_target[hard_example_indice].numel() == 0:
            return pos_target.new_tensor(0.0)

        location_val = iou[hard_example_indice]

#         pos_cls_sigmoid = torch.sigmoid(val) 
#         pos_cls_softmax = F.softmax(pos_cls[hard_example_indice],dim=1)
        val, labels = torch.max(pos_cls[hard_example_indice], dim=1)
        target = torch.sigmoid(val) *0.2
        target = torch.clamp(target,min=1e-7, max=1-1e-7)


#         print(torch.sigmoid(location_val)  )
     
#         target = target * (1- target)

        pred = torch.sigmoid(pos_unknown_pred[hard_example_indice])
        pred = torch.clamp(pred.squeeze(dim=1), min=1e-7, max=1-1e-7)
#         target = 0.65*torch.ones_like(pred)
        loss = - target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
#         loss = - target * torch.log(pred)

        #         print(loss)
        loss = loss.mean()


        return loss if not torch.isnan(loss) else pred.new_tensor(0.0)
    
    def get_up_loss2(self, pos_cls, pos_unknown_pred,iou):
        """
        pos_target: [pos_anchor,num_classes]
        """

        # 1. Max entropy
        pos_cls_sigmoid = torch.sigmoid(pos_cls)
        
        pos_entropy = (- torch.mul(pos_cls_sigmoid, torch.log(pos_cls_sigmoid))).sum(1)  # [pos_num] 每一个正样本的交叉熵
        
        sorted, max_entropy_indices = torch.sort(pos_entropy, descending=True)  # 降序排序
       
        max_entropy_indices = max_entropy_indices[:30]  # [10]
    
        hard_example_indice = max_entropy_indices

        scores = pos_cls_sigmoid[hard_example_indice]  # hard_num,num_classes
        n,m = scores.shape
        
        # 正样本 最大类别概率和对应的标签
           
        # 如果没有正样本
        if n == 0:
            print("无正样本")
            return pos_unknown_pred.new_tensor(0.0)
        
#         location_val = iou[hard_example_indice]

        val, labels = torch.max(scores, dim=1)

        target = val * 0.6
    
#         target = target * (1- target)
#         print(torch.sigmoid(val))
#         print(torch.sigmoid(location_val)  )
#         print(target)
#         target = target * (1- target)

        pred = torch.sigmoid(pos_unknown_pred[hard_example_indice])
        pred = torch.clamp(pred.squeeze(dim=1), min=1e-7, max=1-1e-7)
        
        loss = - target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    
        #         print(loss)
        loss = loss.mean()
      
        return loss

    def get_ic_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                           cls_preds_per_image):
        pair_wise_ious = self.bboxes_iou(bboxes_preds_per_image, gt_bboxes_per_image, False)
        iou, gt_idx = torch.max(pair_wise_ious, dim=1, keepdim=False)
        gt_classes = gt_classes[gt_idx]

        return iou, gt_classes

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                        cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
        # -------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        # -------------------------------------------------------#
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # -------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        # -------------------------------------------------------#
        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # -------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        # -------------------------------------------------------#
        if self.fp16:
            with torch.cuda.amp.autocast(enabled=False):
                cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(
                    0).repeat(num_gt, 1, 1).sigmoid_()
                gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(
                    1, num_in_boxes_anchor, 1)
                pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(
                    -1)
        else:
            cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(
                0).repeat(num_gt, 1, 1).sigmoid_()
            gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1,
                                                                                                                   num_in_boxes_anchor,
                                                                                                                   1)
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
            del cls_preds_

        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost,
                                                                                                       pair_wise_ious,
                                                                                                       gt_classes,
                                                                                                       num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
                          center_radius=2.5):
        # -------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        # -------------------------------------------------------#
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        # -------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        # -------------------------------------------------------#
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1,
                                                                                                                  total_num_anchors)

        # -------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # -------------------------------------------------------#
        #   is_in_boxes     [num_gt, n_anchors_all]
        #   is_in_boxes_all [n_anchors_all]
        # -------------------------------------------------------#
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(
            0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1,
                                                                                total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(
            0)

        # -------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        # -------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        # -------------------------------------------------------#
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # -------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        # -------------------------------------------------------#
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # -------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        # -------------------------------------------------------#
        matching_matrix = torch.zeros_like(cost)

        # ------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        # ------------------------------------------------------------#
        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            # ------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            # ------------------------------------------------------------#
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # ------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        # ------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            # ------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            # ------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        # ------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        # ------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        # ------------------------------------------------------------#
        #   对fg_mask进行更新
        # ------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # ------------------------------------------------------------#
        #   获得特征点对应的物品种类
        # ------------------------------------------------------------#
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain=0.02):
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


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
