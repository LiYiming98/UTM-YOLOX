# 特征图可视化

import matplotlib.pyplot as plt
from torchvision import transforms
import os
import torch
import numpy as np
import math


# 手动修改：feature_num的值修改，为三个特征层的通道数，因为我用的yolox-s，则分别为128/256/512
def feature_visualization(features, feature_num=512):  # 128/256/512，特征图通道数
    """
    features: The feature map which you need to visualization
    model_type: The type of feature map
    model_id: The id of feature map
    feature_num: The amount of visualization you need
    """
    # 分特征层进行特征图像保存，此处为路径
    save_dir = "feat_out/features_{}/".format(feature_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # features[数字]，数字要修改，其中数字代表特征层，0-128,1-256,2-512
    features = torch.tensor([item.cpu().detach().numpy() for item in features[2]])  # 其中的[数字]代表特征图层 0-128,1-256,2-512
    print(features.shape)
    # block by channel dimension
    blocks = torch.chunk(features, features.shape[1], dim=1)

    # 4张特征图为一个图片进行输出，其中range(数字)，数字要进行修改，因为我用yolox-s，则分别为128/4=32,256/4=64,512/4=128
    for j in range(128):  # range(数字)需要适应性修改
        plt.figure()
        for i in range(4):
            torch.squeeze(blocks[4 * j + i])
            feature = transforms.ToPILImage()(blocks[i].squeeze())
            ax = plt.subplot(int(2), int(2), i + 1)
            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(feature)

        plt.savefig(save_dir + 'yolox_feature_map_{}.png'
                    .format(j), dpi=300)