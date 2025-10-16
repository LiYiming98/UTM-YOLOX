import os

import cv2
import cv2 as cv
import xml.etree.ElementTree as ET
import torch
from PIL import ImageDraw, ImageFont
from numpy import mean
from pandas import np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

def iou(gt_box,pred_box,iou_thresh=0.4):
    """
    第一步: 求相交的面积 s
    第二步：求并的面积：s3 = s1+s2-s
    第三步： IoU = s/s3
    """
    gt_top = gt_box[0]
    gt_left = gt_box[1]
    gt_bottom = gt_box[2]
    gt_right = gt_box[3]
    gt_w = gt_right - gt_left
    gt_h = gt_bottom - gt_top


    pred_top = pred_box[0]
    pred_left = pred_box[1]
    pred_bottom = pred_box[2]
    pred_right = pred_box[3]
    pred_w = pred_right - pred_left
    pred_h = pred_bottom - pred_top

    # 交集计算：
    # 交集的w = min(gt_right,pred_right) - max(gt_left,pred_left)
    # h = min(gt_bottom,pred_bottom) - max(gt_top,pred_top)
    s = 0
    if pred_right < gt_left or gt_right < pred_left or pred_bottom < gt_top or gt_bottom<pred_top:
        s = 0.0
    else:
        s_w = min(gt_right,pred_right) - max(gt_left,pred_left)
        s_h = min(gt_bottom,pred_bottom) - max(gt_top,pred_top)
        s = float(s_w * s_h)

    gt_s = gt_w * gt_h
    pred_s = pred_w * pred_h
    return s/(gt_s+pred_s-s) >= iou_thresh


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / torch.clamp(b1_area + b2_area - inter_area, min=1e-6)

    return iou


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):

    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)  # bboxes_a的面积：w*h
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)  # bboxes_b的面积
    else:
        # cx,c,y,w,h -> t l b r
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



def xml_jpg2labelled(imgs_path,xmls_path):
    # os.listdir:返回指定文件夹中包含的文件和文件夹的名字的列表
    imgs_list = os.listdir(imgs_path)
    xmls_list = os.listdir(xmls_path)
    count = 0

    # len(list):返回一个列表的长度
    nums = len(imgs_list)
    print(nums)
    label = []

    for i in range(nums):
        # os.path.join:路径拼接
        img_path = os.path.join(imgs_path,imgs_list[i])

        xml_path = os.path.join(xmls_path,xmls_list[i])
        # img = cv.imread(img_path)

        img = cv.imread(img_path)
        labelled = img
        # ET.parse(xml_path):将xml文件解析为ElementTree
        # 然后从ElementTree的根结点往下搜索
        root = ET.parse(xml_path).getroot()
        objects = root.findall("object")
        label_per_image = []
        for obj in objects:
            bbox = obj.find("bndbox")

            # bbox.find("xmin")返回Element("xmin")
            # .text:Element 类中的属性text
            # .strip()：去除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            # float():类型转换
            name = obj.find("name").text
            # if name == 'Dredger' or name == 'cell-container' :
            if name == 'Dredger' :
                name = 'unknown'
            xmin = int(float(bbox.find("xmin").text.strip()))
            ymin = int(float(bbox.find("ymin").text.strip()))
            xmax = int(float(bbox.find("xmax").text.strip()))
            ymax = int(float(bbox.find("ymax").text.strip()))
            label_per_image.append([name,xmin,ymin,xmax,ymax])
            # print([name,ymin,xmin,ymax,xmax])
           # thickness = int(max((img.size[0] + img.size[1]) // np.mean(input_shape), 3))
           # draw = ImageDraw.Draw(img)
           #  for i in range(3):
           #     labelled = cv.rectangle(labelled,(xmin+i,ymin+i),(xmax-i,ymax-i),(0,255,0),i)
           #     # draw.rectangle([xmin + i, ymin + i, xmax - i, ymax - i], outline=self.colors[c])
           #  cv.imwrite(img_path, labelled)
            count += 1
        # label：所有图片的gt [img_count,gt_bbox_count,top,left,bottom,right]
        label.append(label_per_image)
        # label_per_image=[]

    return label
# 原始图片路径
imgs_path = "D:/mogaidaima/Open-detection/retinaNet/retinanet-pytorch-master/test_code/test_imgs/No_cut/JPEGImages"
# imgs_path = "D:/mogaidaima/Open-detection/retinaNet/retinanet-pytorch-master/test_code/test_imgs/JPEGImages"
# label路径
# xmls_path = "D:/mogaidaima/Open-detection/retinaNet/retinanet-pytorch-master/test_code/test_imgs/Annotations"
xmls_path = "D:/mogaidaima/Open-detection/retinaNet/retinanet-pytorch-master/test_code/test_imgs/No_cut/Annotations"
predictions_out_path = "D:/mogaidaima/open_yolox_sar/yolox-pytorch-main/prediction_out_test"
imgs_out_path = "D:/mogaidaima/open_yolox_sar/yolox-pytorch-main/img_out"

# ep180-loss6.535-val_loss5.906.pth 1024 1024 0.54 0.5
# 每张图中的预测框坐标

# pred_path = "D:/object_detection/yolox-pytorch-main/prediction_out/4.txt"
# file = open(pred_path,"r")
# list=[]
# line = file.readline()
# while line:
#     top = int(float(line.strip().split(" ")[0]))
#     left = int(float(line.strip().split(" ")[1]))
#     bottom = int(float(line.strip().split(" ")[2]))
#     right = int(float(line.strip().split(" ")[3]))
#     line = file.readline()
#     list.append([top,left,bottom,right])
# print(list)
# print(float(file.readline().strip().split(" ")[0]))


def get_res(imgs_path,xmls_path,predictions_out_path,num_classes,iou_thresh = 0.3):
    classes =[
        'ore-oil',
               'Container',
               'Fishing',
               'cell-container',
               'LawEnforce',
        # 'Dredger',
               'unknown'
               ]
    labels = xml_jpg2labelled(imgs_path, xmls_path)
    preds_list = os.listdir(predictions_out_path)
    preds = []

    pred_cnt_all = 0 # 预测框总数
    gt_cnt_all = 0 # gt总数

    imgs_list = os.listdir(imgs_path)

    pred_cls_cnt = [0 for i in range(num_classes)] # 每一类的预测框
    gt_cls_cnt = [0 for i in range(num_classes)] # 每一类的gt


    # xxx = 0

    """
    1. 把每一张图的预测框存到 preds 中
    """
    for i in range(len(preds_list)):
        preds_per_image = []
        pred_path = os.path.join(predictions_out_path, preds_list[i])
        file = open(pred_path, "r")
        line = file.readline()
        while line:
            name = line.strip().split(" ")[0]
            left = int(float(line.strip().split(" ")[1]))
            top = int(float(line.strip().split(" ")[2]))
            right = int(float(line.strip().split(" ")[3]))
            bottom = int(float(line.strip().split(" ")[4]))
            score = round(float(line.strip().split(" ")[5]),3)
            unknown_score = round(float(line.strip().split(" ")[6]),3)
            line = file.readline()

            preds_per_image.append([name, top, left, bottom, right,score,unknown_score])
        preds.append(preds_per_image)

    """
    2. 统计 每一类的预测框数量，gt数量，预测框总数，gt总数
    """
    for i in range(len(labels)):
        label_per_image = labels[i]
        pred_per_image = preds[i]
        pred_cnt_all += len(pred_per_image)
        gt_cnt_all += len(label_per_image)

        # 统计每一类的预测框数量
        for j in range(len(pred_per_image)):
            pred_box_cls = pred_per_image[j][0]
            print(pred_box_cls)
            pred_cls_cnt[classes.index(pred_box_cls)] += 1

        # 统计每一类的gt数量
        for k in range(len(label_per_image)):
            gt_cls = label_per_image[k][0]
            gt_cls_cnt[classes.index(gt_cls)] += 1


    """
    3. 遍历每一张测试图
    """

    xujing_cnt = [0 for i in range(num_classes)]
    tp_cnt = [0 for i in range(num_classes)]
    class_error = [[0 for i in range(num_classes)] for i in range(num_classes)]
    tp_no_cls = [0 for i in range(num_classes)]


    tp_all_cnt = 0
    tp_shibie = 0

    # 已知类目标的未知类输出
    list_known_unknown_out = []

    # 未知类的未知类输出
    list_unknown_out = []

    for i in range(len(labels)):
        img_path = os.path.join(imgs_path, imgs_list[i])
        image_out_path = os.path.join(imgs_out_path, imgs_list[i])
        img = cv.imread(img_path)
        labelled = img

        label_per_image = labels[i]
        pred_per_image = preds[i]


        pred = []
        pred_cls = []
        pred_score = []

        pred_unknown_score = []
        gt = []
        gt_cls = []

        for k in range(len(pred_per_image)):
            pred.append(pred_per_image[k][1:5])
            pred_cls.append(pred_per_image[k][0])
            pred_score.append(pred_per_image[k][5])
            pred_unknown_score.append(pred_per_image[k][6])
            print(pred_per_image[k][5])

        for k in range(len(label_per_image)):
            gt.append(label_per_image[k][1:5])
            gt_cls.append(label_per_image[k][0])





        pred_per_image_box = torch.tensor(pred)
        pred_per_image_cls = pred_cls
        pred_per_image_score = pred_score
        pred_per_image_unknown_score = pred_unknown_score

        label_per_image_box = torch.tensor(gt)
        label_per_image_cls = gt_cls


        if(pred_per_image_box.shape[0] == 0):
            continue
        pred_to_gt_ious = bboxes_iou(pred_per_image_box,label_per_image_box) # [pred_all,gt_all]
        gt_to_pred_ious = bboxes_iou(label_per_image_box,pred_per_image_box) # [gt_all,pred_all]



        val0,idx0 = torch.max(pred_to_gt_ious,dim=1)
        val1,idx1 = torch.max(gt_to_pred_ious,dim=1)

        length,_ = pred_to_gt_ious.shape

        length2,_ = gt_to_pred_ious.shape

        # 遍历每一个预测框
        for j in range(length):
            # 如果预测框 和gt的 iou 小于阈值，那么是背景杂波
            if val0[j] < iou_thresh:
                xujing_cnt[classes.index(pred_per_image_cls[j])] += 1

                if val0[j] < iou_thresh:

                    # if(pred_per_image_cls[j] == "unknown") :
                        # 未知类虚警
                    if (pred_per_image_cls[j] == "unknown"):
                        list_unknown_out.append(pred_per_image_unknown_score[j])
                    else:
                        # 已知类虚警
                        list_known_unknown_out.append(pred_per_image_unknown_score[j])

                    xujing_cnt[classes.index(pred_per_image_cls[j])] += 1
                    for m in range(3):
                        ymin = int(pred_per_image_box[j][1])
                        xmin = int(pred_per_image_box[j][0])
                        ymax = int(pred_per_image_box[j][3])
                        xmax = int(pred_per_image_box[j][2])

                        # 虚警

                        labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (0, 0, 255), m)
                    cv2.putText(labelled, str(pred_per_image_cls[j]) +":"+ str(pred_per_image_score[j]) ,
                                (int(pred_per_image_box[j][0]), int(pred_per_image_box[j][1]) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                2)
            # 第j个预测框和哪一个gt匹配，然后再看与该gt匹配的预测框是不是j
            elif idx1[idx0[j]] != j: # idx0[j] : 这个预测框匹配的gt的索引，idx1[idx0[j]]：匹配的gt 与哪个预测框最匹配
                xujing_cnt[classes.index(pred_per_image_cls[j])] += 1
                # if str(pred_per_image_cls[j]) == "unknown":
                #     for m in range(3):
                #         ymin = int(pred_per_image_box[j][1])
                #         xmin = int(pred_per_image_box[j][0])
                #         ymax = int(pred_per_image_box[j][3])
                #         xmax = int(pred_per_image_box[j][2])
                #         # 虚警
                #         labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (255, 255, 255), m)
                #     cv2.putText(labelled, str(pred_per_image_cls[j]),
                #                 (int(pred_per_image_box[j][0]), int(pred_per_image_box[j][1]) - 12),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                #                 2)
                # else:
                if (label_per_image_cls[idx0[j]] == "unknown"):
                    # 未知类虚警

                    list_unknown_out.append(pred_per_image_unknown_score[j])
                else:
                    # 已知类目标 错判 为 其他类
                    list_known_unknown_out.append(pred_per_image_unknown_score[j])


                for m in range(3):
                    ymin = int(pred_per_image_box[j][1])
                    xmin = int(pred_per_image_box[j][0])
                    ymax = int(pred_per_image_box[j][3])
                    xmax = int(pred_per_image_box[j][2])
                    # 虚警
                    labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (0, 0, 255), m)
                cv2.putText(labelled, str(pred_per_image_cls[j])+":"+str(pred_per_image_score[j]),
                            (int(pred_per_image_box[j][0]), int(pred_per_image_box[j][1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)

            # 类比相同时，为gt
            elif label_per_image_cls[idx0[j]] == pred_per_image_cls[j]:
                tp_cnt[classes.index(pred_per_image_cls[j])] += 1
                tp_no_cls[classes.index(pred_per_image_cls[j])] += 1
                tp_all_cnt += 1
                tp_shibie += 1

                if (pred_per_image_cls[j] == "unknown"):
                    # 未知类正确检测识别

                    list_unknown_out.append(pred_per_image_unknown_score[j])
                else:
                    # 已知类正确检测识别
                    list_known_unknown_out.append(pred_per_image_unknown_score[j])

                if str(pred_per_image_cls[j]) == "unknown":
                    for m in range(3):
                        ymin = int(pred_per_image_box[j][1])
                        xmin = int(pred_per_image_box[j][0])
                        ymax = int(pred_per_image_box[j][3])
                        xmax = int(pred_per_image_box[j][2])
                        # 紫色
                        labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (255, 0, 255), m)
                    cv2.putText(labelled, str(pred_per_image_cls[j]) +":" + str(pred_per_image_score[j]),
                                (int(pred_per_image_box[j][0]), int(pred_per_image_box[j][1]) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                                2)
                else:
                    for m in range(3):

                        ymin = int(pred_per_image_box[j][1])
                        xmin = int(pred_per_image_box[j][0])
                        ymax = int(pred_per_image_box[j][3])
                        xmax = int(pred_per_image_box[j][2])
                        # gt 绿色
                        labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (0,255,0), m)
                    cv2.putText(labelled, str(pred_per_image_cls[j]) +":"+ str(pred_per_image_score[j]), (int(pred_per_image_box[j][0]) , int(pred_per_image_box[j][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),
                                    2)
            else:
                # 最匹配的gt的类别，实际判成的类别
                class_error[classes.index(label_per_image_cls[idx0[j]])][classes.index(pred_per_image_cls[j])] += 1
                tp_no_cls[classes.index(label_per_image_cls[idx0[j]])] += 1
                tp_all_cnt += 1

                # 按gt区分
                if (label_per_image_cls[idx0[j]] == "unknown"):
                    list_unknown_out.append(pred_per_image_unknown_score[j])
                else:
                    list_known_unknown_out.append(pred_per_image_unknown_score[j])

                if str(pred_per_image_cls[j]) == "unknown":
                    for m in range(3):
                        ymin = int(pred_per_image_box[j][1])
                        xmin = int(pred_per_image_box[j][0])
                        ymax = int(pred_per_image_box[j][3])
                        xmax = int(pred_per_image_box[j][2])
                        # 虚警
                        labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (255, 255, 255), m)
                    cv2.putText(labelled, str(pred_per_image_cls[j]) +":"+ str(pred_per_image_score[j]),
                                (int(pred_per_image_box[j][0]), int(pred_per_image_box[j][1]) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2)
                else:
                    for m in range(3):
                        ymin = int(pred_per_image_box[j][1])
                        xmin = int(pred_per_image_box[j][0])
                        ymax = int(pred_per_image_box[j][3])
                        xmax = int(pred_per_image_box[j][2])
                        # 虚警
                        labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (255, 255, 255), m)
                    cv2.putText(labelled, str(pred_per_image_cls[j]) +":"+ str(pred_per_image_score[j]), (int(pred_per_image_box[j][0]) , int(pred_per_image_box[j][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2)
        # print("未知类数量" + str(len(list_unknown_out)))
        # print(x)
        # print(y)
        x = set()

        for j in range(length2):

            if val1[j] <= iou_thresh:

                for m in range(3):

                    ymin = int(label_per_image_box[j][1])
                    xmin = int(label_per_image_box[j][0])
                    ymax = int(label_per_image_box[j][3])
                    xmax = int(label_per_image_box[j][2])

                    # tp
                    labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (255,0,0), m)
                cv2.putText(labelled, label_per_image_cls[j], (int(label_per_image_box[j][0]) , int(label_per_image_box[j][1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2)
            elif str(idx1[j]) in x:

                for m in range(3):
                    ymin = int(label_per_image_box[j][1])
                    xmin = int(label_per_image_box[j][0])
                    ymax = int(label_per_image_box[j][3])
                    xmax = int(label_per_image_box[j][2])

                    # tp
                    labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m), (255, 0, 0), m)
                cv2.putText(labelled, label_per_image_cls[j],
                            (int(label_per_image_box[j][0]), int(label_per_image_box[j][1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2)
            else:

                x.add(str(idx1[j]))
        cv.imwrite(image_out_path, labelled)


    data1 = np.array(list_known_unknown_out)
    data2 = np.array(list_unknown_out)

    plt.rcParams['font.sans-serif'] = ['SimSun']

    plt.rcParams['axes.unicode_minus'] = False

    plt.hist(x=[data1,data2],  # 绘图数据
             bins=20,  # 指定直方图的条形数为20个
             edgecolor='w',  # 指定直方图的边框色
             color=['c', 'r'],  # 指定直方图的填充色
             label=['已知类', '未知类'],  # 为直方图呈现图例
             density=False,  # 是否将纵轴设置为密度，即频率
             alpha=0.6,  # 透明度
             rwidth=1,  # 直方图宽度百分比：0-1
             stacked=False,
             )  # 当有多个数据时，是否需要将直方图呈堆叠摆放，默认水平摆放

    plt.title("未知类判别分支输出统计直方图", fontsize=18)
    plt.xlabel('未知类判别概率', fontsize=16)
    plt.ylabel('目标数量', fontsize=16)

    ax = plt.gca()  # 获取当前子图
    ax.spines['right'].set_color('none')  # 右边框设置无色
    ax.spines['top'].set_color('none')  # 上边框设置无色
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()




    print("预测框总数: " + str(pred_cnt_all) )
    print("gt总数: " + str(gt_cnt_all))
    print("tp总数:" + str(tp_all_cnt) )
    print("检测率:" + str(tp_all_cnt / gt_cnt_all))
    precision_all = tp_all_cnt / pred_cnt_all
    recall_all = tp_all_cnt / gt_cnt_all
    f1 = precision_all * recall_all * 2 / (precision_all + recall_all)
    print("虚警率:" + str(1 - (tp_all_cnt / pred_cnt_all)))

    print("识别率:" + str(tp_shibie / tp_all_cnt))
    print("Precision_all: " + str(precision_all))
    print("Recall_all:" +str(recall_all))
    print("F1:" + str(f1))

    f = open("D:/mogaidaima/Open-detection/retinaNet/retinanet-pytorch-master/test_code/res/result.txt","w")
    f.write("预测框总数: " + str(pred_cnt_all) + "\n")
    f.write("gt总数: " + str(gt_cnt_all) + "\n")
    f.write("检测率:" + str(tp_all_cnt / gt_cnt_all) + "\n")
    f.write("虚警率:" + str(1 - tp_all_cnt / pred_cnt_all) + "\n")

    f.write("识别率:" + str(tp_shibie / tp_all_cnt) + "\n")
    f.write("\n")


    pred_all_without_unknown = 0
    class_error_unknown = 0


    tp_cls = [] # 每一类的tp数
    pred_cls = [] # 每一类的预测数
    gt_cls = [] # 每一类的gt数

    for i in range(len(classes)):
        if i != 5:
            pred_all_without_unknown += pred_cls_cnt[i]
            class_error_unknown += class_error[i][5]
        f.write("类：" + classes[i] + "\n")
        f.write("类无关的检测数量：" + str(tp_no_cls[i]) + "   ")
        f.write("tp：" + str(tp_cnt[i]) + "   ")
        f.write("pred：" + str(pred_cls_cnt[i]) + "   ")
        f.write("gt:" + str(gt_cls_cnt[i]) + "   ")
        f.write("虚警：" + str(xujing_cnt[i]) + "   ")
        f.write("漏警:" + str(gt_cls_cnt[i] - tp_no_cls[i]) + "   ")
        f.write("\n")

        print("%s: 类无关的检测数量=%d, TP=%d,pred= %d,GT=%d;漏警=%d" % (classes[i],tp_no_cls[i],
                                                               tp_cnt[i], pred_cls_cnt[i],gt_cls_cnt[i],gt_cls_cnt[i] - tp_no_cls[i]))
        for j in range(len(classes)):
            print("%s: %d  " % (classes[j], class_error[i][j]))

    print("WI:" + str(class_error_unknown/pred_all_without_unknown))
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for i in range(len(classes)):
        precision = tp_cnt[i] / (pred_cls_cnt[i] * 1.0)
        precision_sum += precision

        recall = tp_cnt[i] / (gt_cls_cnt[i] * 1.0)
        recall_sum += recall
        f1_score = 2 * precision * recall/(precision + recall)
        f1_sum += f1_score
        print("%s: precision=%s,recall=%s,f1-score=%s" %
              (classes[i],
               str(precision),
               str(recall),
               str(f1_score)
              ))

    print("平均: precision=%s,recall=%s,f1-score=%s" %
              (
               str(precision_sum / 6),
               str(recall_sum / 6),
               str(f1_sum / 6)
              ))






















def draw_predtion(imgs_path,xmls_path,predictions_out_path,imgs_out_path,num_classes):
    """
    Precision: TP/(TP+FP) = TP/Pred_count
    Recall: TP/(TP+FN) = TP/Gt_count

    """
    classes = ['ore-oil',
                   'Container',
                   'Fishing',
                   'cell-container',
                   'LawEnforce',
                    # 'Dredger',
                   'unknown']


    labels = xml_jpg2labelled(imgs_path,xmls_path)
    preds_list = os.listdir(predictions_out_path)
    preds=[]


    TP_all = 0
    pred_cnt_all = 0
    gt_cnt_all = 0

    TP_cnt = [0 for i in range(num_classes)] # 类相关的TP
    TP_cnt_only_dect = [0 for i in range(num_classes)]
    pred_cnt = [0 for i in range(num_classes)]
    gt_cnt = [0 for i in range(num_classes)]

    class_error = [[0 for i in range(num_classes)] for i in range(num_classes)]
    xujing_cls = [0 for i in range(num_classes)] # 每一类的虚警个数
    """
    把每一张图的预测框存到preds中
    """
    for i in range(len(preds_list)):
        preds_per_image = []
        pred_path = os.path.join(predictions_out_path,preds_list[i])
        file = open(pred_path,"r")
        line = file.readline()
        while line:
            name = line.strip().split(" ")[0]
            left = int(float(line.strip().split(" ")[1]))
            top = int(float(line.strip().split(" ")[2]))
            right = int(float(line.strip().split(" ")[3]))
            bottom = int(float(line.strip().split(" ")[4]))
            line = file.readline()
            preds_per_image.append([name,top, left, bottom, right])
        preds.append(preds_per_image)
        # preds_per_image=[]

    pred_tag = []
    label_tag = []
    for i in range(len(labels)):
        TP_Per_image_Boxes = []
        label_per_image = labels[i]
        pred_per_image = preds[i]
        pred_cnt_all += len(pred_per_image)

        pred_per_image_tag = [0 for _ in range(len(pred_per_image))]
        label_per_image_tag = [0 for _ in range(len(label_per_image))]
        gt_cnt_all += len(label_per_image)

        # 统计每一类的预测框数量
        for j in range(len(pred_per_image)):
            pred_box = pred_per_image[j][1:5]
            pred_box_cls = pred_per_image[j][0]
            pred_cnt[classes.index(pred_box_cls)] += 1

        # 统计每一类的gt数量
        for k in range(len(label_per_image)):
            gt_box = label_per_image[k][1:5]

            gt_cls = label_per_image[k][0]
            gt_cnt[classes.index(gt_cls)] += 1

        # 对每张图片的每个预测框，看预测框是否与某个GT匹配，若匹配，则为TP
        biaoji = [0 for p in range(len(pred_per_image))]
        biaoji2 = [0 for p in range(len(pred_per_image))]
        gt_biaoji = [0 for p in range(len(label_per_image))]





        for j in range(len(pred_per_image)):
            pred_box = pred_per_image[j][1:5]
            pred_box_cls = pred_per_image[j][0]

            matched = False

            for k in range(len(label_per_image)):
                gt_box = label_per_image[k][1:5]
                gt_cls = label_per_image[k][0]

                # 若该预测框与某个gt的IOU>0.5则为TP
                if iou(gt_box,pred_box):
                    # 加入TP集合
                    # 每一类的tp数量
                    # 类无关的tp
                    matched = True
                    if biaoji[j] == 0:
                        # 如果这个gt此时还没有任何预测框和它匹配，那么当前预测框成为tp
                        # 每一个gt只和一个预测框进行匹配，其他匹配的都视作虚警

                        TP_all += 1
                        biaoji[j] = 1
                        TP_cnt_only_dect[classes.index(gt_cls)] += 1

                    if pred_box_cls == gt_cls:
                        if biaoji2[j] == 0:
                            TP_cnt[classes.index(gt_cls)] += 1
                            biaoji2[j] = 1
                        pred_per_image_tag[j] = 1
                        label_per_image_tag[k] = 1
                    else:
                        class_error[classes.index(gt_cls)][classes.index(pred_box_cls)] += 1

            if not matched:
                xujing_cls[classes.index(pred_box_cls)] += 1


                    # continue
        pred_tag.append(pred_per_image_tag)
        label_tag.append(label_per_image_tag)

    for i in range(len(classes)):
        print("%s预测数量：%d" % (classes[i], pred_cnt[i]))
        print("%sgt数量：%d" % (classes[i], gt_cnt[i]))
        print("%s tp数量 ： %d" % (classes[i], TP_cnt[i]))

    ############画图并计算Precision和Recall#####################
    imgs_list = os.listdir(imgs_path)
    TP=[]
    Prediction=[]
    Ground_truth=[]
    for i in range(len(imgs_list)):
        image_path = os.path.join(imgs_path,imgs_list[i])
        image_out_path = os.path.join(imgs_out_path,imgs_list[i])
        img = cv.imread(image_path)
        labelled = img
        pred_pre_image_tag = pred_tag[i]
        label_pre_image_tag = label_tag[i]

        # thickness = int(max((img.size[0] + img.size[1]) // np.mean(input_shape), 3))

        # draw = ImageDraw.Draw(img)
        count = 0 # 记录TP数量
        for j in range(len(pred_pre_image_tag)):
            # 画出TP
            if pred_pre_image_tag[j] == 1:
                count += 1
                for m in range(2):
                    ymin = preds[i][j][1]
                    xmin = preds[i][j][2]
                    ymax = preds[i][j][3]
                    xmax = preds[i][j][4]
                    labelled = cv.rectangle(labelled, (ymin + m, xmin + m), (ymax - m, xmax - m), (0,255,0), m)
                    # draw.rectangle([xmin + i, ymin + i, xmax - i, ymax - i], outline=(0,255,0))
            else:
            # 画出虚警
                for m in range(2):
                    ymin = preds[i][j][1]
                    xmin = preds[i][j][2]
                    ymax = preds[i][j][3]
                    xmax = preds[i][j][4]
                    labelled = cv.rectangle(labelled, (ymin + m, xmin + m), (ymax - m, xmax - m),(255, 0, 0), m)
        TP.append(float(count))
        Prediction.append(len(pred_pre_image_tag))
        Ground_truth.append(len(label_pre_image_tag))

        print(count)
        print(len(pred_pre_image_tag))
        print(len(label_pre_image_tag))

        for k in range(len(label_pre_image_tag)):
            if label_pre_image_tag[k] == 0:
                # 画出漏警
                for m in range(3):
                    xmin = labels[i][k][1]
                    ymin = labels[i][k][2]
                    xmax = labels[i][k][3]
                    ymax = labels[i][k][4]

                    labelled = cv.rectangle(labelled, (xmin + m, ymin + m), (xmax - m, ymax - m),(0, 0, 255), m)

        cv.imwrite(image_out_path,labelled)

    print("tp_all:" + str(TP_all))
    print("gt_cnt_all:" + str(gt_cnt_all))
    print("pred_cnt_all" + str(pred_cnt_all))
    print(TP_all / pred_cnt_all)
    print(TP_all / gt_cnt_all)
    return TP,Prediction,Ground_truth,TP_cnt,pred_cnt,gt_cnt,class_error,TP_cnt_only_dect,xujing_cls

def print_result():
    TP,Prediction,Ground_truth,TP_cnt,pred_cnt,gt_cnt,class_error,TP_cnt_only_dect,xujing_cls = draw_predtion(imgs_path,xmls_path,predictions_out_path,imgs_out_path,6)
    Precision = sum(TP) / sum(Prediction)
    Recall = sum(TP) / sum(Ground_truth)
    F1_score = 2 * Precision * Recall / (Precision+Recall)
    print("无关类别时，F1=%f ;Recall=%f;Precision= %f" % (F1_score,Recall,Precision))
    # 'ore-oil',
    # 'Container',
    # 'Fishing',
    # 'cell-container',
    # 'LawEnforce',
    # 'unknown'

    classes = ['ore-oil',
                   'Container',
                   'Fishing',
                   'cell-container',
                   'LawEnforce',
                   'unknown']

    for i in range(len(classes)):
        if pred_cnt[i] != 0:
            Precision = TP_cnt[i] / pred_cnt[i]
        Recall = TP_cnt[i] / gt_cnt[i]
        F1_score = 2 * Precision * Recall / (Precision + Recall)

        print("%s: F1=%f ;Recall=%f;Precision= %f" % (classes[i],F1_score,Recall,Precision))
        print("%s: TP_不关心识别=%d,TP_识别正确=%d,pred= %d,GT=%d;虚警=%d" % (classes[i], TP_cnt_only_dect[i],TP_cnt[i],pred_cnt[i], gt_cnt[i],xujing_cls[i]))
        for j in range(len(classes)):
            print("%s: %d  " %(classes[j],class_error[i][j]))








# print_result()

get_res(imgs_path,xmls_path,predictions_out_path,6)




