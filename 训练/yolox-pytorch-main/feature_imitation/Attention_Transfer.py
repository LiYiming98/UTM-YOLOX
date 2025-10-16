import torch
from torch import nn
from torch.nn import functional as F
class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer
    文章: https://arxiv.org/pdf/1612.03928.pdf
    """
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        """
        通过平均池化将教师网络和学生网络的shape对齐,然后计算MSE loss
        """
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

if __name__ == "__main__":
    at = Attention()
    s_3 = torch.rand([2,40,40,40])
    s_4 = torch.rand([2,112,20,20])
    s_5 = torch.rand([2,160,40,40])
    s = (s_3,s_4,s_5)
    t_3 = torch.rand([2, 320, 40, 40])
    t_4 = torch.rand([2, 640, 20, 20])
    t_5 = torch.rand([2, 1280, 40, 40])
    t = (t_3, t_4, t_5)
    at_loss = at.forward(s,t)
    print(sum(at_loss))




