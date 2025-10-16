import torch
from torch import nn
import torch.nn.functional as F

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=14):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        b,all_anchor,c = x.size() # torch.Size([4, 1600, 68])
        x = F.softmax(x.reshape(-1,self.reg_max+1),dim=1) # [4*1600*4,17]
        x_pre_jifen = x.reshape([b,all_anchor,4,self.reg_max+1])  # 每个预测框的tlbr表示的概率分布,用于计算DFL

        x = F.linear(x, self.project.type_as(x)).reshape([b,all_anchor,4])
        return x,x_pre_jifen

