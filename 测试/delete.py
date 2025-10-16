import torch

from thop import profile
import torch.nn as nn
from ptflops import get_model_complexity_info
# from nets.yolo_mobileNetV3_small_dfl import YoloBody
# from nets.yolo_yuanshi import YoloBody_yuanshi as YoloBody
from nets.yolo_tiny import YoloBody
# from nets.yolo_yuanshi_dfl import YoloBody_yuanshi_dfl as YoloBody
# from nets.yolo_mobileNetV3_small import YoloBody
from utils.utils import get_classes
import re


'''
训练的时候使用的MobileNetv3_small中没有去掉用于分类的三个层： conv,avgpool,classifier
因此这里把这些多余的层删除

delete的时候，把mobileNet.mobvilenet_v3_small中注释掉的输出模块添加回去，测试的时候再注释掉
'''
# device = torch.device("cpu")
# classes_path    = 'model_data/close.txt'
# phi = 's'
# class_names, num_classes = get_classes(classes_path)
# model1 = YoloBody(num_classes, phi)
# model1.load_state_dict(torch.load('logs/mobile/ep200-loss4.268-val_loss5.557.pth'))
#
# print(model1)
# C1 = nn.Sequential()
# C2 = nn.Sequential()
# C3 = nn.Sequential()
# model1.backbone.backbone.model.conv = C1
# model1.backbone.backbone.model.avgpool = C2
# model1.backbone.backbone.model.classifier = C3
# torch.save(model1.state_dict(),'logs/mobile/ep200-loss4.268-val_loss5.557_cut.pth')
# print(model1)

device = torch.device("cpu")
classes_path    = 'model_data/close.txt'
phi = 'tiny'
class_names, num_classes = get_classes(classes_path)
print(num_classes)
model1 = YoloBody(num_classes, phi)
model1.load_state_dict(torch.load('logs/mobile/tiny/ep175-loss3.743-val_loss5.001.pth'))

macs, params = get_model_complexity_info(model1, (3, 512, 512), as_strings=True, print_per_layer_stat=True)


# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<16}'.format(params))
print('FLOPS:  ' + flops)
print('Params: ' + params)
