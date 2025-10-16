import torch
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np
from collections import deque

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import random

class memory_bank():
    """
    初始化： 已知类个数 和 队列的最大长度
    """

    def __init__(self, num_known_classes, ins_con_loss_weight, max_iter, ins_con_in_queue_size=128):
        # 初始化一个列表，为每一个已知类创建一个双端队列
        # ins_con_in_queue_size: 队列长度
        self.num_known_classes = num_known_classes
        self.bank = torch.zeros([num_known_classes, ins_con_in_queue_size, 128]).to('cuda')

        # memorybank 中每一个queue的ptr, 这个ptr表示的是，下一次从哪个位置开始入队，
        self.bank_queue_ptr = torch.zeros([num_known_classes, 1])
        self.bank_queue_label = torch.zeros([num_known_classes, ins_con_in_queue_size, 1]).to('cuda') - 1

        self.num_known_classes = num_known_classes
        self.ins_con_in_queue_size = ins_con_in_queue_size
        self.sample_num = 32
        self.ins_con_loss_weight = ins_con_loss_weight  # IC loss的权重
        self.max_iter = max_iter
        self.current_iter = 0

        self.tau = 0.1

        self.mse = torch.nn.MSELoss(reduction="sum")
        self.prototype = []
        self.pro_cls = []
        self.p = 0.5 # 0.7

        self.last_epoch = 0
        self.last_epoch1 = 0
        self.last_epoch2 = 0
        self.loss = torch.tensor([0.0])
        self.hingeloss = nn.HingeEmbeddingLoss(35)

    # feat : [所有正样本数,128]  gt_classes : [所有正样本数，对应的类别index]
    # IC输出中 对应某个类的所有的正样本：  singleClass_positive_IC_outputs [某个类的正样本数量，128]
    # 以及这个张量对应的类别

    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        texts = ['ore-oil','Container','Fishing','cell-container','LawEnforce']
        
        m0 = []
        n0 = []
        
        m1 = []
        n1 = []
        
        m2 = []
        n2 = []
        
        m3 = []
        n3 = []
        
        m4 = []
        n4 = []
        for i in range(data.shape[0]):
            if label[i] == 0:
                m0.append(data[i, 0])
                n0.append(data[i, 1])
            if label[i] == 1:
                m1.append(data[i, 0])
                n1.append(data[i, 1])
            if label[i] == 2:
                m2.append(data[i, 0])
                n2.append(data[i, 1])
            if label[i] == 3:
                m3.append(data[i, 0])
                n3.append(data[i, 1])
            if label[i] == 4:
                m4.append(data[i, 0])
                n4.append(data[i, 1])
                
#             plt.scatter(data[i, 0], data[i, 1], marker=".", color=colors[int(label[i])],label=texts[int(label[i])])
        #             plt.text(data[i, 0], data[i, 1], str(label[i]),
        #                      color= colors[int(label[i])],
        #                      fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(m0, n0, marker=".", color=colors[0],label=texts[0])
        plt.scatter(m1, n1, marker=".", color=colors[1],label=texts[1])
        plt.scatter(m2, n2, marker=".", color=colors[2],label=texts[2])
        plt.scatter(m3, n3, marker=".", color=colors[3],label=texts[3])
        plt.scatter(m4, n4, marker=".", color=colors[4],label=texts[4])

        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig

    def _dequeue_and_enqueue(self, feat, gt_classes, ious, iou_thr=0.7):
        """
        在计算IC loss前先完成入队和出队

        feat : [batch_size * 2100，128]
        gt_classes:  [batch_size * 2100,1] 每个点与所有gt对应的类别
        ious : [batch_size * 2100] 每个点与所有gt的iou的最大值


        """
     
        with torch.no_grad():
            ious_mask = ious > iou_thr  # 根据阈值和类别筛选出需要的样本
            feat_positive = feat[ious_mask.squeeze(), :]  # [num_pos,128]

            gt_classes = gt_classes[ious_mask.squeeze()]  # [num_pos,1]
         
            # 获得该类对应的队列
            for i in range(self.num_known_classes):
                ptr = int(self.bank_queue_ptr[i])  # memorybank 中每一个queue的ptr, 这个ptr表示的是，下一次从哪个位置开始入队，

                # self.bank[i]: [ins_con_in_queue_size,128]
                # self.bank_queue_label[i][:,0] != -1 即 找出 [0,ins_con_in_queue_size)中标签不为-1的张量，标签为-1说明还没有存进去，是空张量，要剔除
                cls_queue = self.bank[i][self.bank_queue_label[i][:, 0] != -1]  # [队列中已经存在的张量数,128]

                # cls_ind ：在[0,num_pos)中找出属于第i类的idx [ 正样本中第i类样本的 mask ]
                cls_ind = (gt_classes == i).squeeze(1)
                # cls_feat: [第i类正样本的数量, 128]
                # cls_gt_classes : [第i类正样本的数量, 1]
               
                cls_feat, cls_gt_classes = feat_positive[cls_ind], gt_classes[cls_ind]  # 取出正样本中所有属于第i类的张量

                # 该类的所有正样本张量和 队列中张量计算余弦相似度, cls_feat: IC头输出张量 ， cls_queue:memory bank中的张量
                _, sim_inds = F.cosine_similarity(
                    cls_feat.unsqueeze(1), cls_queue.unsqueeze(0), dim=-1).mean(dim=1).sort()
                # 采样 64 个
                top_sim_inds = sim_inds[:self.sample_num]  # 取出前self.ins_con_in_queue_size 最不相似的张量的idx
                cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]

                # 如果 ptr + cls_feat.size(0) <= self.ins_con_queue_size ,那么把 cls_feat全部入队
                # 否则，只把队列补满： self.ins_con_queue_size - ptr
                batch_size = cls_feat.size(
                    0) if ptr + cls_feat.size(0) <= self.ins_con_in_queue_size else self.ins_con_in_queue_size - ptr
                # 入队

                self.bank[i][ptr:ptr + batch_size] = cls_feat[:batch_size]
                self.bank_queue_label[i][ptr:ptr +
                                             batch_size] = cls_gt_classes[:batch_size]
                # 如果队不满，则继续入队，否则，从头开始覆盖

                ptr = ptr + batch_size if ptr + batch_size < self.ins_con_in_queue_size else 0

                self.bank_queue_ptr[i] = ptr
                
    def _dequeue_and_enqueue2(self, feat, gt_classes, ious, iou_thr=0.7):
        """
        在计算IC loss前先完成入队和出队

        feat : [batch_size * 2100，128]
        gt_classes:  [batch_size * 2100,1] 每个点与所有gt对应的类别
        ious : [batch_size * 2100] 每个点与所有gt的iou的最大值


        """
     
        with torch.no_grad():
            ious_mask = ious > iou_thr  # 根据阈值和类别筛选出需要的样本
            feat_positive = feat[ious_mask.squeeze(), :]  # [num_pos,128]

            gt_classes = gt_classes[ious_mask.squeeze()]  # [num_pos,1]
         
            # 获得该类对应的队列
            for i in range(self.num_known_classes):
                ptr = int(self.bank_queue_ptr[i])  # memorybank 中每一个queue的ptr, 这个ptr表示的是，下一次从哪个位置开始入队，

                # self.bank[i]: [ins_con_in_queue_size,128]
                # self.bank_queue_label[i][:,0] != -1 即 找出 [0,ins_con_in_queue_size)中标签不为-1的张量，标签为-1说明还没有存进去，是空张量，要剔除
                cls_queue = self.bank[i][self.bank_queue_label[i][:, 0] != -1]  # [队列中已经存在的张量数,128]

                # cls_ind ：在[0,num_pos)中找出属于第i类的idx [ 正样本中第i类样本的 mask ]
                cls_ind = (gt_classes == i).squeeze(1)
                # cls_feat: [第i类正样本的数量, 128]
                # cls_gt_classes : [第i类正样本的数量, 1]
               
                cls_feat, cls_gt_classes = feat_positive[cls_ind], gt_classes[cls_ind]  # 取出正样本中所有属于第i类的张量

                # 该类的所有正样本张量和 队列中张量计算余弦相似度, cls_feat: IC头输出张量 ， cls_queue:memory bank中的张量
                
                N,_ = cls_feat.shape
                S= N
                if N >= self.sample_num:
                    S = self.sample_num
                
                index = torch.LongTensor(random.sample(range(N), S)).to(cls_feat.device)
                
                cls_feat = torch.index_select(cls_feat, 0, index).to(cls_feat.device)
                cls_gt_classes = torch.index_select(cls_gt_classes, 0, index).to(cls_feat.device)
#                 cls_feat, cls_gt_classes = cls_feat[top_sim_inds], cls_gt_classes[top_sim_inds]

                # 如果 ptr + cls_feat.size(0) <= self.ins_con_queue_size ,那么把 cls_feat全部入队
                # 否则，只把队列补满： self.ins_con_queue_size - ptr
                batch_size = cls_feat.size(
                    0) if ptr + cls_feat.size(0) <= self.ins_con_in_queue_size else self.ins_con_in_queue_size - ptr
                # 入队

                self.bank[i][ptr:ptr + batch_size] = cls_feat[:batch_size]
                self.bank_queue_label[i][ptr:ptr +
                                             batch_size] = cls_gt_classes[:batch_size]
                # 如果队不满，则继续入队，否则，从头开始覆盖

                ptr = ptr + batch_size if ptr + batch_size < self.ins_con_in_queue_size else 0

                self.bank_queue_ptr[i] = ptr
                
    def Proto_loss_2(self,feat_positive,gt_classes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Prototypes = torch.cat(self.prototype, dim=0).to(device)  # [5,256]
        Prototypes_cls = torch.cat(self.pro_cls, dim=0).to(device)  # [5,1]
        dist = torch.cdist(feat_positive, Prototypes,p=2)  # [pos_num,5]
        labels = []
        
        n,_ = feat_positive.shape
        for i in range(n):
            for j in range(5):
                if gt_classes[i][0] == Prototypes_cls[j][0]:
                    labels.append(1)
                else:
                    labels.append(-1)
        
        loss = self.hingeloss(dist,torch.tensor(labels).reshape(-1,5).cuda())
        
   
        return loss  if not torch.isnan(loss) else feat_positive.new_tensor(0.0)

    def Prototype_IC_loss(self, feat, gt_classes, ious,oln_preds,  epoch,iou_thr=0.5):
        """
        epoch = 20时，计算原型
        之后每5轮更新一次原型
        """
        self._dequeue_and_enqueue2(feat, gt_classes, ious)

        if epoch >= 30 and epoch % 5 == 0 and self.last_epoch != epoch and (not torch.isnan(self.loss)):
            self.last_epoch = epoch
            queues = []
            labels = []
            for i in range(self.num_known_classes):
                p = self.prototype[i]
                p = p.unsqueeze(dim=0)
                feats = self.bank[i,:,:]
                label = self.bank_queue_label[i,:,:]
                dist = torch.cdist(feats, p).squeeze()  
                val,idx = torch.topk(dist,64,largest=False)
                queues.append(feats[idx])
                labels.append(label[idx])
            
            queue = torch.cat(queues,dim=0).reshape(-1,128).detach().cpu().numpy()
            label = torch.cat(labels,dim=0).reshape(-1).detach().cpu().numpy()
                
            
#             queue = self.bank.reshape(-1, 128).detach().cpu().numpy()
#             label = self.bank_queue_label.reshape(-1).detach().cpu().numpy()
            
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time()
            result = tsne.fit_transform(queue)
            fig = self.plot_embedding(result, label,
                                      't-SNE embedding of the digits (time %.2fs)'
                                      % (time() - t0))
            plt.savefig('./nets/tsne/result' + str(epoch) + '.jpg')

            plt.close()


        if epoch == 25:
            queue = self.bank.reshape(-1, 128)  # [num_known_classes * ins_con_in_queue_size，128]
            queue_label = self.bank_queue_label.reshape(-1)  # [num_known_classes * ins_con_in_queue_size]

            queue_inds = queue_label != -1
            queue, queue_label = queue[queue_inds], queue_label[queue_inds]
            if self.last_epoch1 != epoch:
                for i in range(self.num_known_classes):
                    cur_cls_tensor = queue_label == i
                    count_true = torch.sum(cur_cls_tensor).item()
                    
                    if count_true == 0:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        yuanxing_x = torch.zeros([1,128]).to(device)
                    else:
                        yuanxing_x = torch.mean(queue[cur_cls_tensor], dim=0).unsqueeze(dim=0)  # [128]
                    self.prototype.append(yuanxing_x)
                    self.pro_cls.append(torch.tensor([[i]]))
                self.last_epoch1 = epoch

            pos_inds = ious >= iou_thr
            feat_positive = feat[pos_inds.squeeze(), :]  # [正样本数，128]
            gt_classes = gt_classes[pos_inds.squeeze()]

            decay_weight = 1.0 - epoch / self.max_iter
            loss = self.Proto_loss_2(feat_positive, gt_classes) * 0.5 * decay_weight

            return loss
        #             return self.Proto_loss(feat_positive, gt_classes) * 0.1 * decay_weight

        elif epoch >25:

            if epoch % 5 == 0 and self.last_epoch2 != epoch:
                queue = self.bank.reshape(-1, 128)  # [num_known_classes * ins_con_in_queue_size，128]
                queue_label = self.bank_queue_label.reshape(-1)  # [num_known_classes * ins_con_in_queue_size]
                queue_inds = queue_label != -1
                queue, queue_label = queue[queue_inds], queue_label[queue_inds]
                for i in range(self.num_known_classes):
                    cur_cls_tensor = queue_label == i
                    count_true = torch.sum(cur_cls_tensor).item()
                    
                    if count_true == 0:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        yuanxing_x = torch.zeros([1,128]).to(device)
                    else:
                        yuanxing_x = torch.mean(queue[cur_cls_tensor], dim=0).unsqueeze(dim=0)  # [128]
#                     yuanxing_x = torch.mean(queue[cur_cls_tensor], dim=0).unsqueeze(dim=0)  # [128]
                    self.prototype[i] = self.p * self.prototype[i] + (1 - self.p) * yuanxing_x
                self.last_epoch2 = epoch

            pos_inds = ious >= iou_thr
            feat_positive = feat[pos_inds.squeeze(), :]  # [正样本数，128]
            gt_classes = gt_classes[pos_inds.squeeze()]
            
            ######### 未知类拉远
#             neg_ids = ious <= 0.2
#             idx2 = oln_preds >= 0.65
#             neg_ids = torch.logical_and(idx2.squeeze(), neg_ids.squeeze())
#             feat_neg = feat[neg_ids,:]
#             loss_neg= self.Proto_loss_3(feat_neg) * 0.1 

            decay_weight = 1.0 - epoch / self.max_iter
            loss = self.Proto_loss_2(feat_positive, gt_classes) * 0.5
#             decay_weight = 1.0 - epoch / self.max_iter
#             decay_weight = 1.0
#             loss = self.Proto_loss_2(feat_positive, gt_classes) * 0.5 * decay_weight

            return loss
        #             return self.Proto_loss(feat_positive, gt_classes) * 0.1 * decay_weight

        else:
            return feat.new_tensor(0.0)

    def Proto_loss(self, feat_positive, gt_classes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if feat_positive.shape[0] == 0:
            return feat_positive.new_tensor(0.0)

        Prototypes = torch.cat(self.prototype, dim=0).to(device)  # [5,256]
        Prototypes_cls = torch.cat(self.pro_cls, dim=0).to(device)  # [5,1]
        # torch.dist(input_1,input_2,p=2)
        dist = torch.cdist(feat_positive, Prototypes)  # [pos_num,5]
#         dist = dist / torch.clamp(dist.norm(), 0.0001, 100)
        #         dist = torch.div(
        #             torch.matmul(feat_positive,Prototypes.T),
        #             0.1)  # [pos_num,memoryBank中所有非空张量] / tau,第i行表示 第i个正样本和所有memorybank中非空张量的点积
        
        mask = torch.eq(gt_classes.unsqueeze(dim=1), Prototypes_cls.T).int()  # [pos_num,5]
    
        mask_No = (~(mask.bool())).int()
        match_dist = dist * mask
    
        
#         a,b = torch.max(dist * mask_No,dim=1)
#         print(torch.max(a))
        not_match_dist = (25 - dist) * mask_No
        not_match_dist = torch.where(not_match_dist < 0, torch.zeros_like(not_match_dist), not_match_dist)
        

        
        
        # 如果是余弦相似度作为距离
        # loss1 = torch.sum((1 - dist) * mask,dim=1)
        # not_match_dist = (0.5 - dist) * mask_No
        # not_match_dist = torch.where(not_match_dist < 0, torch.zeros_like(not_match_dist), not_match_dist)
        # loss = (loss1 + torch.sum(not_match_dist,dim=1)).sum()

        loss = match_dist.mean() + not_match_dist.mean()
  
        n = feat_positive.shape[0]
     
        self.loss = loss

        return loss * 0.1 if not torch.isnan(loss) else feat_positive.new_tensor(0.0)
    
    
    def Proto_loss_3(self,feat_neg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Prototypes = torch.cat(self.prototype, dim=0).to(device)  # [5,256]
        Prototypes_cls = torch.cat(self.pro_cls, dim=0).to(device)  # [5,1]
        dist = torch.cdist(feat_neg, Prototypes,p=2)  # [pos_num,5]
        print(torch.max(torch.max(dist,dim=1)))
        labels = []
        
        n,_ = feat_neg.shape
        for i in range(n):
            for j in range(5):
                    labels.append(-1)
        
        loss = self.hingeloss(dist,torch.tensor(labels).reshape(-1,5).cuda())
        
   
        return loss  if not torch.isnan(loss) else feat_neg.new_tensor(0.0)

    def ProtoNCE(self, feat_positive, gt_classes, queue, queue_label, feat_negative):
        Proto_C = []
        cls = []

        device = torch.device("cuda:0")
        for i in range(self.num_known_classes):

            cur_cls_tensor = queue_label == i

            if sum(cur_cls_tensor) == 0:
                continue
            yuanxing_x = torch.mean(queue[cur_cls_tensor], dim=0).unsqueeze(dim=0)  # [128]

            Proto_C.append(yuanxing_x)
            cls.append(torch.tensor([[i]]))

        Proto_C = torch.cat(Proto_C, dim=0)  # [原型个数,128]
        Proto_cls = torch.cat(cls, dim=0).to(device)  # [原型个数，1]

        # 计算正样本的原型对比学习损失
        loss_pos = feat_positive.new_tensor(0.0)
        if feat_positive.shape[0] != 0:
            mask = torch.eq(gt_classes, Proto_cls.T).float()  # [pos_num,原型]

            anchor_dot_contrast = torch.div(
                torch.matmul(feat_positive, Proto_C.T),
                self.tau)  # [pos_num,原型个数] / tau,第i行表示 第i个正样本和所有memorybank中非空张量的点积

            # .detach(): 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
            logits = anchor_dot_contrast

            # exp_logits = torch.exp(logits)
            exp_logits = torch.exp(logits)  # 每一个正样本和所有原型的点积的exp

            log_prob = torch.log((torch.exp(logits) * mask / exp_logits.sum(1, keepdims=True)).sum(dim=1))
            mean_log_prob_pos = (log_prob).mean()  # [pos_num,1]

            loss_pos = - mean_log_prob_pos  # 总的loss取平均

            loss_pos = loss_pos if not torch.isnan(loss_pos) else feat_positive.new_tensor(0.0)

        # 求背景类的ProtoNCE loss
        anchor_dot_contrast_neg = torch.div(
            torch.matmul(feat_negative, Proto_C.T),
            self.tau)  # [neg_num,原型个数] / tau,第i行表示 第i个正样本和所有memorybank中非空张量的点积
        exp_logits = 1 / (torch.exp(anchor_dot_contrast_neg).sum(1))
        loss_neg = - torch.log(exp_logits).mean()
        print(loss_neg)

        return loss_pos + loss_neg

    def get_IC_loss(self, feat, gt_classes, ious, epoch, iou_thr=0.5):
        """
            ious: 21504
            gt_classes: 21504,1
            feat: 21504,128

        """

        self._dequeue_and_enqueue(feat, gt_classes, ious)
        pos_inds = ious >= iou_thr
        
        if not pos_inds.sum():
            return feat.new_tensor(0.0)  # 没有正样本直接返回0
        
        pos_inds = torch.reshape(pos_inds,[-1,1])
        
        
        if epoch >= 15 and epoch % 5 == 0 and self.last_epoch != epoch:
            self.last_epoch = epoch
            queue = self.bank.reshape(-1, 128).detach().cpu().numpy()
            label = self.bank_queue_label.reshape(-1).detach().cpu().numpy()
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            t0 = time()
            result = tsne.fit_transform(queue)
            fig = self.plot_embedding(result, label,
                                      't-SNE embedding of the digits (time %.2fs)'
                                      % (time() - t0))
            plt.savefig('./nets/tsne/result' + str(epoch) + '.jpg')

            plt.close()

        feat_positive = feat[pos_inds.squeeze(), :]  # [正样本数，128]
        gt_classes = gt_classes[pos_inds.squeeze()]  # [num_pos,1]
        
      
        
        queue = self.bank.reshape(-1, 128)  # [num_known_classes * ins_con_in_queue_size，128]
       
        queue_label = self.bank_queue_label.reshape(-1)  # [num_known_classes * ins_con_in_queue_size]
        queue_inds = queue_label != -1
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]
#         n,_ = feat_positive.shape
#         if n >= 300:
#             n = 300
#         feat_positive = feat_positive[:n,:]
#         gt_classes = gt_classes[:n,:]
        loss_ins_con = self.ins_con_loss(feat_positive, gt_classes, queue, queue_label)

        # queues = []
        # queue_labels = []
        # for i in range(self.num_known_classes):
        #     # 把每个队列中存在的张量取出来
        #     queue = self.bank[i]  # [self.ins_con_in_queue_size,128]
        #     queue_label = self.bank_queue_label[i]  # [self.ins_con_in_queue_size,1]
        #     queue_idx = queue_label != -1  # -1 表示这个张量不存在，因此去掉不存在的
        #     queue = queue[queue_idx.squeeze()]
        #     queue_label = queue_label[queue_idx.squeeze()]
        #     queues.append(queue)
        #     queue_labels.append(queue_label)
        # # print(queues[0].shape)
        # # print(queues[1].shape)
        # # print(queues[2].shape)
        #
        # queues = torch.cat(tuple(queues), dim=0)  # [memoryBank中所有张量,128]
        # queue_labels = torch.cat(tuple(queue_labels), dim=0)  # [memoryBank中所有张量,1]

        # queues = torch.stack(queues,dim=0) # [num_knows_classes, bank中该类所有非空的张量数,128]
        # queue_labels = torch.stack(queue_labels,dim=0) # [num_knows_classes, bank中该类所有非空的张量数,1]
        # loss_ins_con = self.ins_con_loss(feat_positive, gt_classes, queues, queue_labels)
        decay_weight = 1.0 - epoch / self.max_iter

        return loss_ins_con * self.ins_con_loss_weight * decay_weight

    def ins_con_loss(self, feat_positive, gt_classes, queues, queue_labels):
        """
        feat_positive: [pos_nums,128]
        gt_classes: [pos_nums,1]
        queues : [memoryBank中所有非空张量，128]
        queue_labels: [memoryBank中所有非空张量，1]
        """
        self.tau = 0.1
        # mask中第i行表示： 第i个正样本 和 memoryBank中所有张量的类别进行比较，相同为1，不同为0
        
        mask = torch.eq(gt_classes, queue_labels.T).float()  # [pos_num,memorybank中所有非空张量]
        
        anchor_dot_contrast = torch.div(
            torch.matmul(feat_positive, queues.T),
            self.tau)  # [pos_num,memoryBank中所有非空张量] / tau,第i行表示 第i个正样本和所有memorybank中非空张量的点积
        
        # dim = 1,即取每一行的最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        # .detach(): 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
        logits = anchor_dot_contrast - logits_max.detach()
        
        logits_mask = torch.ones_like(logits)

        logits_mask[logits == 0] = 0  # 最大值位置 mask = 0
        
        # mask中第i行表示的是第i个正样本和哪些memorybank中的张量的类别一致，而 logits_mask是把 第i个正样本和所有memorybank中非空张量的点积中最大的那个张量的点积置为0
        # 因此乘完之后，还是 第i个正样本和哪些memorybank中的张量的类别一致，但是需要把自己置为0
        mask = mask * logits_mask

        # exp_logits = torch.exp(logits)
        exp_logits = torch.exp(logits) * logits_mask  # 每一行表示一个 正样本 和 所有 memorybank中非空张量的点积，并去除最大值
        
        # log(exp(zi*zj/T) - log(sum(exp(zi*zk / T)       # [pos_num,q_alls]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdims=True))
        
        # print(torch.log(exp_logits.sum(1,keepdims=True)).shape)
        # log_prob此时第i行中的第j个位置的值为：  (zi * zj) / T - log(sum(exp(zi*zk/T)) , zi: 第i个正样本, zj: queue中的张量
        # mask * log_prob ： 每一行取出和 zi类别相同的queue中的张量的值
        # 先按列求和，再取平均，就得到 Lic(zi),即这个正样本算出来的 IC loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # [pos_num,1]

        loss = - mean_log_prob_pos.mean()  # 总的loss取平均
      
       
        return loss if not torch.isnan(loss) else feat_positive.new_tensor(0.0)