import torch.nn as nn
import torch.nn.functional as F
import torch

class Feature_separate_Loss(nn.Module):
    """PyTorch version of `Masked Generative Distillation`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        name (str): the loss name of the layer
        alpha_mgd (float, optional): Weight of dis_loss. Defaults to 0.00002
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 ):
        super(Feature_separate_Loss, self).__init__()


        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.generation_S = nn.Sequential(
            nn.Conv2d(student_channels, student_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(student_channels//2, student_channels//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(student_channels//4,2,kernel_size=3,padding=1),
            nn.ReLU(inplace=True))

        self.generation_T = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // 2, teacher_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels // 4, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))



    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        loss = self.get_dis_loss(preds_S, preds_T)

        return loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_CE = nn.KLDivLoss(reduction="batchmean")
        N, C, H, W = preds_T.shape
        device = preds_S.device

        fea_separate_S = self.generation_S(preds_S)
        fea_separate_T = self.generation_T(preds_T)



        dis_loss = loss_CE(F.log_softmax((fea_separate_S).view(-1,2),dim = 1),
                                     F.softmax((fea_separate_T).view(-1,2),dim = 1))

        return dis_loss

if __name__ == "__main__":
    module = Feature_separate_Loss(64,320)
    s = torch.rand([4,64,40,40])
    t = torch.rand([4,320,40,40])
    dis_loss = module(s,t)
    print(dis_loss)