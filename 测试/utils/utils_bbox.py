import numpy as np
import torch
from torchvision.ops import nms, boxes
# from torchvision.ops import nms


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def decode_outputs(outputs, input_shape):
    grids   = []
    strides = []
    hw      = [x.shape[-2:] for x in outputs]
    #---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    #---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    #---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    #---------------------------------------------------#
    outputs[:, :, 4] = torch.sigmoid(outputs[:, :, 4])
    outputs[:, :, 5:] = torch.sigmoid(outputs[:, :, 5:])


    for h, w in hw:
        #---------------------------#
        #   根据特征层的高宽生成网格点
        #---------------------------#   
        grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
        #---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #---------------------------#   
        grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape           = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    #---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    #---------------------------#
    grids               = torch.cat(grids, dim=1).type(outputs.type())
    strides             = torch.cat(strides, dim=1).type(outputs.type())
    #------------------------#
    #   根据网格点进行解码
    #------------------------#
    outputs[..., :2]    = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides
    #-----------------#
    #   归一化
    #-----------------#
    outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]
    outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]
    return outputs

def non_max_suppression(prediction, oln_out,unknown_out,ic_output,feats,labels,num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner          = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    
    output = [None for _ in range(len(prediction))]
    #----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        # print(unknown_out.shape)
        unknown_out = torch.sigmoid(torch.cat(unknown_out,dim=1))

        oln_out = torch.cat(oln_out,dim=1)
        # idx = oln_out[0,:] >= 0.75
        # oln_out[0,idx] = 1
        ic_output = torch.cat(ic_output,dim=1)
        # oln_out = torch.sigmoid(oln_out)

        unknown = (unknown_out[0,:].squeeze() * torch.clamp(oln_out[0,:].squeeze()*0.9,0,1)).unsqueeze(dim=1)
        # oln_out[0,:]*0.65
        image_pred = torch.cat([image_pred, unknown],dim=1)
        image_pred_clone = image_pred.clone()
        # image_pred = torch.cat([image_pred, unknown], dim=1)
        for j in range(5):
            if j == 1:
                # image_pred[:, 5 + j] = image_pred[:, 5 + j]
                image_pred[:,5 + j] = image_pred[:,5 + j]* image_pred[:, 4]
                # image_pred[:,5 + j] = torch.clamp(image_pred[:,5 + j],0,0.65)
                # image_pred[:, 5 + j] = image_pred[:, 5 + j]* oln_out[0,:].squeeze()
            else:
                image_pred[:, 5 + j] = image_pred[:, 5 + j] * image_pred[:, 4]
                # image_pred[:, 5 + j] = image_pred[:, 5 + j]
                # image_pred[:, 5 + j] = image_pred[:, 5 + j]  * oln_out[0,:].squeeze()
            # image_pred[:, 5 + j] = image_pred[:, 5 + j] * oln_out[0,:].squeeze()
        image_pred2 = image_pred.clone()
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        for j in range(5):
            image_pred_clone[:, 5 + j] = image_pred_clone[:, 5 + j]* oln_out[0,:].squeeze()
        class_conf3,_ = torch.max(image_pred_clone[:, 5:5 + num_classes-1], 1)
        class_obj = image_pred_clone[:,4]

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#

        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        # conf_mask1 = (class_conf[:, 0] >= 0.6).squeeze()
        # conf_mask = (oln_out[0, :, 0] >= 0.65).squeeze()
        # conf_mask = conf_mask & conf_mask1
        # class_pred2 = torch.ones_like(class_conf)


        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        class_pred2 = torch.ones_like(class_conf)
        oln_out = oln_out[0, conf_mask, :]
        # unknown_out = unknown[conf_mask]
        unknown_out = unknown_out[0,conf_mask,:]
        ic_out = ic_output[0,conf_mask,:]
        print(ic_out.shape)
        class_conf3 = class_conf3[conf_mask]
        class_obj = class_obj[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred2.float()), 1)
        # detections = detections[conf_mask]
        # print("class_conf:" + str(detections[5]))
        detections_class = detections[detections[:, -1] == 1]
        keep = nms(
            detections_class[:, :4],
            detections_class[:, 4],
            nms_thres
        )
        print("oln_out" + str(oln_out[keep]))
        print("unknown" + str(unknown_out[keep]))
        print("class_conf3" + str(class_conf3[keep]))
        print("class_obj" + str(class_obj[keep]))


        detections_class = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float(),unknown_out.float()), 1)
        max_detections = detections_class[keep]
        print(max_detections[:,4])
        print(keep.shape)
        output[i]   = max_detections if output[i] is None else torch.cat((output[i], max_detections))



        ic_out = ic_out[keep]
        class_pred = class_pred[keep]
        feats.append(ic_out)
        labels.append(class_pred)
        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]
            
        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data
            
        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
        
        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output


def non_max_suppression_2unknown(prediction, oln_out, unknown_out, ic_output, feats, labels, num_classes, input_shape,
                        image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # ----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        # print(unknown_out.shape)
        unknown_out = torch.sigmoid(torch.cat(unknown_out, dim=1))

        oln_out = torch.cat(oln_out, dim=1)
        # idx = oln_out[0,:] >= 0.75
        # oln_out[0,idx] = 1
        ic_output = torch.cat(ic_output, dim=1)
        # oln_out = torch.sigmoid(oln_out)

        unknown = (unknown_out[0, :].squeeze() * torch.clamp(oln_out[0, :].squeeze(), 0, 1)).unsqueeze(dim=1)
        # oln_out[0,:]*0.65
        image_pred = torch.cat([image_pred, unknown], dim=1)
        image_pred_clone = image_pred.clone()
        # image_pred = torch.cat([image_pred, unknown], dim=1)
        for j in range(4):
            if j == 1:
                image_pred[:, 5 + j] = image_pred[:, 5 + j] * image_pred[:, 4]
                # image_pred[:,5 + j] = torch.clamp(image_pred[:,5 + j],0,0.65)
                # image_pred[:, 5 + j] = image_pred[:, 5 + j]* oln_out[0,:].squeeze()
            else:
                image_pred[:, 5 + j] = image_pred[:, 5 + j] * image_pred[:, 4]
                # image_pred[:, 5 + j] = image_pred[:, 5 + j]  * oln_out[0,:].squeeze()
            # image_pred[:, 5 + j] = image_pred[:, 5 + j] * oln_out[0,:].squeeze()
        image_pred2 = image_pred.clone()
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        for j in range(5):
            image_pred_clone[:, 5 + j] = image_pred_clone[:, 5 + j] * oln_out[0, :].squeeze()
        class_conf3, _ = torch.max(image_pred_clone[:, 5:5 + num_classes - 1], 1)
        class_obj = image_pred_clone[:, 4]

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#

        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        # conf_mask1 = (class_conf[:, 0] >= 0.6).squeeze()
        # conf_mask = (oln_out[0, :, 0] >= 0.7).squeeze()
        # conf_mask = conf_mask & conf_mask1
        # class_pred2 = torch.ones_like(class_conf)

        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        class_pred2 = torch.ones_like(class_conf)
        oln_out = oln_out[0, conf_mask, :]
        # unknown_out = unknown[conf_mask]
        unknown_out = unknown_out[0, conf_mask, :]
        ic_out = ic_output[0, conf_mask, :]
        print(ic_out.shape)
        class_conf3 = class_conf3[conf_mask]
        class_obj = class_obj[conf_mask]
        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred2.float()), 1)
        # detections = detections[conf_mask]
        # print("class_conf:" + str(detections[5]))
        detections_class = detections[detections[:, -1] == 1]
        keep = nms(
            detections_class[:, :4],
            detections_class[:, 4],
            nms_thres
        )
        print("oln_out" + str(oln_out[keep]))
        print("unknown" + str(unknown_out[keep]))
        print("class_conf3" + str(class_conf3[keep]))
        print("class_obj" + str(class_obj[keep]))

        detections_class = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float(), unknown_out.float()),
                                     1)
        max_detections = detections_class[keep]
        print(max_detections[:, 4])
        print(keep.shape)
        output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        ic_out = ic_out[keep]
        class_pred = class_pred[keep]
        feats.append(ic_out)
        labels.append(class_pred)
        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]

        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data

        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output


def non_max_suppression_center(prediction, oln_out, unknown_out, ic_output, center_out,feats, labels, num_classes, input_shape,
                        image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    # ----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        # print(unknown_out.shape)
        unknown_out = torch.sigmoid(torch.cat(unknown_out, dim=1))

        oln_out0 = torch.cat(oln_out, dim=1)
        center_out = torch.cat(center_out,dim=1)

        oln_out = torch.sqrt(oln_out0 * center_out)


        ic_output = torch.cat(ic_output, dim=1)
        # oln_out = torch.sigmoid(oln_out)

        unknown = (unknown_out[0, :].squeeze() * torch.clamp(oln_out[0, :].squeeze(), 0, 1)).unsqueeze(dim=1)
        # oln_out[0,:]*0.65
        image_pred = torch.cat([image_pred, oln_out[0, :]], dim=1)
        image_pred_clone = image_pred.clone()
        # image_pred = torch.cat([image_pred, unknown], dim=1)
        for j in range(5):
            image_pred[:, 5 + j] = image_pred[:, 5 + j] * image_pred[:, 4]
            # image_pred[:, 5 + j] = image_pred[:, 5 + j] * oln_out[0,:].squeeze()
        image_pred2 = image_pred.clone()
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        for j in range(5):
            image_pred_clone[:, 5 + j] = image_pred_clone[:, 5 + j]
        class_conf3, _ = torch.max(image_pred_clone[:, 5:5 + num_classes - 1], 1)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#

        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        # conf_mask1 = (class_conf[:, 0] >= 0.5).squeeze()
        # conf_mask = (oln_out[0, :, 0] >= 0.7).squeeze()
        # conf_mask = conf_mask & conf_mask1

        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        class_pred2 = torch.ones_like(class_conf)
        oln_out = oln_out[0, conf_mask, :]

        center_out = center_out[0,conf_mask,:]
        unknown_out = unknown_out[0, conf_mask, :]
        ic_out = ic_output[0, conf_mask, :]
        print(ic_out.shape)
        class_conf3 = class_conf3[conf_mask]
        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred2.float()), 1)
        # detections = detections[conf_mask]
        # print("class_conf:" + str(detections[5]))
        detections_class = detections[detections[:, -1] == 1]
        keep = nms(
            detections_class[:, :4],
            detections_class[:, 4],
            nms_thres
        )
        print("oln_out" + str(oln_out[keep]))
        print("center_out" + str(center_out[keep]))
        print("unknown" + str(unknown_out[keep]))
        print("class_conf3" + str(class_conf3[keep]))
        detections_class = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
        max_detections = detections_class[keep]
        print(max_detections[:, 4])
        print(keep.shape)
        output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        ic_out = ic_out[keep]
        class_pred = class_pred[keep]
        feats.append(ic_out)
        labels.append(class_pred)
        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]

        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data

        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output


def non_max_suppression_close(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner          = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    #----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        print(image_pred.shape)
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        keep_scores, keep_classes, keep_pred_bboxes, keep_true_cls = nms_diou(detections[:, 4],
                                                                              detections[:, 5],
                                                                              detections[:, :4], class_pred[:, 0],
                                                                              nms_thres)
        res = torch.cat([keep_pred_bboxes, keep_scores.unsqueeze(dim=1), keep_true_cls.unsqueeze(dim=1)], dim=1)
        output[i] = res if output[i] is None else torch.cat((output[i], res))
        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )

        # output[i]   = detections[nms_out_index]

        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]

        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data

        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output


def nms_diou(one_image_scores, one_image_classes, one_image_pred_bboxes,true_cls,nms_threshold):
    """
    one_image_scores:[anchor_nums],4:classification predict scores
    one_image_classes:[anchor_nums],class indexes for predict scores
    one_image_pred_bboxes:[anchor_nums,4],4:x_min,y_min,x_max,y_max
    """
    # Sort boxes
    # 按 score进行排序
    sorted_one_image_scores, sorted_one_image_scores_indexes = torch.sort(
        one_image_scores, descending=True)
    # 排序之后的类别
    sorted_one_image_classes = one_image_classes[
        sorted_one_image_scores_indexes]

    sorted_true_cls = true_cls[sorted_one_image_scores_indexes]
    # 排序之后的预测框
    sorted_one_image_pred_bboxes = one_image_pred_bboxes[
        sorted_one_image_scores_indexes]
    # 预测框的w和h
    sorted_pred_bboxes_w_h = sorted_one_image_pred_bboxes[:,
                                                          2:] - sorted_one_image_pred_bboxes[:, :
                                                                                             2]
    # 预测框的 面积 w * h
    sorted_pred_bboxes_areas = sorted_pred_bboxes_w_h[:,
                                                      0] * sorted_pred_bboxes_w_h[:,
                                                                           1]
    # 类别数
    detected_classes = torch.unique(sorted_one_image_classes, sorted=True)

    keep_scores, keep_classes, keep_pred_bboxes,keep_true_cls = [], [], [],[]
    # 对每一个类别
    for detected_class in detected_classes:
        # 找到当前类别的预测框
        single_class_scores = sorted_one_image_scores[
            sorted_one_image_classes == detected_class]

        single_class_pred_bboxes = sorted_one_image_pred_bboxes[
            sorted_one_image_classes == detected_class]
        single_class_pred_bboxes_areas = sorted_pred_bboxes_areas[
            sorted_one_image_classes == detected_class]
        single_class = sorted_one_image_classes[sorted_one_image_classes ==
                                                detected_class]

        single_true_cls = sorted_true_cls[sorted_one_image_classes ==
                                                detected_class]

        single_keep_scores,single_keep_classes,single_keep_pred_bboxes,single_keep_true_cls=[],[],[],[]
        while single_class_scores.numel() > 0:
            # 找 score 最高的预测框
            top1_score, top1_class, top1_pred_bbox,top1_pred_bbox_true_cls = single_class_scores[
                0:1], single_class[0:1], single_class_pred_bboxes[0:1],single_true_cls[0:1]

            single_keep_scores.append(top1_score)
            single_keep_classes.append(top1_class)
            single_keep_pred_bboxes.append(top1_pred_bbox)
            # top1对应的真实类别
            single_keep_true_cls.append(top1_pred_bbox_true_cls)

            # top1预测框的 面积
            top1_areas = single_class_pred_bboxes_areas[0]

            if single_class_scores.numel() == 1:
                break
            # 除去 top1预测框后剩下的 框的 score，class，boxes,area
            single_class_scores = single_class_scores[1:]
            single_class = single_class[1:]
            single_class_pred_bboxes = single_class_pred_bboxes[1:]
            single_class_pred_bboxes_areas = single_class_pred_bboxes_areas[
                1:]
            # 除去 top1预测框后剩下的 框的 真实类别
            single_true_cls = single_true_cls[1:]

            # 和top1计算DIOU
            overlap_area_top_left = torch.max(
                single_class_pred_bboxes[:, :2], top1_pred_bbox[:, :2])
            overlap_area_bot_right = torch.min(
                single_class_pred_bboxes[:, 2:], top1_pred_bbox[:, 2:])
            overlap_area_sizes = torch.clamp(overlap_area_bot_right -
                                             overlap_area_top_left,
                                             min=0)
            overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:,
                                                                         1]

            # compute union_area
            union_area = top1_areas + single_class_pred_bboxes_areas - overlap_area
            union_area = torch.clamp(union_area, min=1e-4)
            # compute ious for top1 pred_bbox and the other pred_bboxes
            ious = overlap_area / union_area

            top1_pred_bbox_ctr = (top1_pred_bbox[:, 2:4] +
                                  top1_pred_bbox[:, 0:2]) / 2
            single_class_pred_bboxes_ctr = (
                single_class_pred_bboxes[:, 2:4] +
                single_class_pred_bboxes[:, 0:2]) / 2
            p2 = (top1_pred_bbox_ctr[:, 0] -
                  single_class_pred_bboxes_ctr[:, 0])**2 + (
                      top1_pred_bbox_ctr[:, 1] -
                      single_class_pred_bboxes_ctr[:, 1])**2

            enclose_area_top_left = torch.min(
                top1_pred_bbox[:, 0:2], single_class_pred_bboxes[:, 0:2])
            enclose_area_bot_right = torch.max(
                top1_pred_bbox[:, 2:4], single_class_pred_bboxes[:, 2:4])
            enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                             enclose_area_top_left,
                                             min=1e-4)
            c2 = (enclose_area_sizes[:, 0])**2 + (enclose_area_sizes[:,
                                                                     1])**2

            dious = ious - p2 / c2

            single_class_scores = single_class_scores[
                dious < nms_threshold]


            single_class = single_class[dious < nms_threshold]
            single_class_pred_bboxes = single_class_pred_bboxes[
                dious < nms_threshold]
            single_class_pred_bboxes_areas = single_class_pred_bboxes_areas[
                dious < nms_threshold]
            single_true_cls = single_true_cls[dious < nms_threshold]

        single_keep_scores = torch.cat(single_keep_scores, axis=0)
        single_keep_classes = torch.cat(single_keep_classes, axis=0)
        single_keep_pred_bboxes = torch.cat(single_keep_pred_bboxes,
                                            axis=0)

        single_keep_true_cls = torch.cat(single_keep_true_cls,axis=0)


        keep_scores.append(single_keep_scores)
        keep_classes.append(single_keep_classes)
        keep_pred_bboxes.append(single_keep_pred_bboxes)
        keep_true_cls.append(single_keep_true_cls)

    keep_scores = torch.cat(keep_scores, axis=0)
    keep_classes = torch.cat(keep_classes, axis=0)
    keep_pred_bboxes = torch.cat(keep_pred_bboxes, axis=0)
    keep_true_cls = torch.cat(keep_true_cls,axis = 0)

    return keep_scores, keep_classes, keep_pred_bboxes,keep_true_cls
def decode_outputs_close(outputs, input_shape):
    grids   = []
    strides = []
    hw      = [x.shape[-2:] for x in outputs]
    #---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    #---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    #---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    #---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        #---------------------------#
        #   根据特征层的高宽生成网格点
        #---------------------------#
        grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
        #---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #---------------------------#
        grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape           = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    #---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    #---------------------------#
    grids               = torch.cat(grids, dim=1).type(outputs.type())
    strides             = torch.cat(strides, dim=1).type(outputs.type())
    #------------------------#
    #   根据网格点进行解码
    #------------------------#
    outputs[..., :2]    = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides
    #-----------------#
    #   归一化
    #-----------------#
    outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]
    outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]
    return outputs