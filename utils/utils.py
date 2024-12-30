from collections import Iterable
import torch.nn as nn
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    def __init__(self, path, int_form=':04d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)

        return log

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], correct.squeeze()


# class Feather_statistic(nn.Module):
#     def __init__(self, confidence_thresh=0.0,T_max = 10,T_k = 5,rightT=0,args=None):
#         super(Feather_statistic, self).__init__()
#         self.Queue_mus_list = []
#         self.Queue_sigmas_list = []
#         self.confidence_thresh = confidence_thresh
#         self.T_max = T_max
#         self.T_k = T_k
#         self.rightT = rightT


#     def append(self,mus,sigmas,Qneue_lenghth = 50000):
#         mus = mus.tolist()
#         sigmas = sigmas.tolist()

#         self.Queue_mus_list.extend(mus)
#         if len(self.Queue_mus_list)> Qneue_lenghth:
#             self.Queue_mus_list = self.Queue_mus_list[-Qneue_lenghth:]

#         self.Queue_sigmas_list.extend(sigmas)
#         if len(self.Queue_sigmas_list) > Qneue_lenghth:
#             self.Queue_sigmas_list = self.Queue_sigmas_list[-Qneue_lenghth:]

#     def forward(self, features, labels,pred,confidence=None,modify_T = False,append=True,return_distance = False):
#         """
#         Args:
#             features: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         T =  torch.ones_like(labels)
#         min_distances = torch.zeros_like(labels)
#         correctly_classified = torch.zeros_like(labels)
#         if confidence!=None:
#             if pred==None:
#                 pred = labels
#             correctly_classified[(labels==pred)&(confidence>self.confidence_thresh)] = 1
            

#             means = torch.mean(features, dim=[1, 2, 3])
#             stds = torch.std(features, dim=[1, 2, 3])
#             if append==True:
#                 self.append(means[correctly_classified==1],stds[correctly_classified==1])
#         if modify_T == True and correctly_classified.sum()!=correctly_classified.numel() and len(self.Queue_mus_list)!=0:
            
#             means = means.unsqueeze(1)  # [num_samples_A, 1]
#             stds = stds.unsqueeze(1)    # [num_samples_A, 1]
#             meansB_expanded = torch.tensor(self.Queue_mus_list).cuda().unsqueeze(0)
#             stdsB_expanded = torch.tensor(self.Queue_sigmas_list).cuda().unsqueeze(0)

#             # 计算Wasserstein距离矩阵
#             wasserstein_distances = torch.sqrt((means - meansB_expanded) ** 2 + (stds - stdsB_expanded) ** 2)

#             # 找到每个样本对应的最小Wasserstein距离
#             min_distances, min_indices = torch.min(wasserstein_distances, dim=1)
#             # import pdb
#             # pdb.set_trace()
#             if self.rightT == 1:
#                 T = 1 + (self.T_max - 1) * torch.exp(self.T_k*min_distances)
#             elif self.rightT == 2:
#                 T = 0.001+ torch.exp(self.T_k*min_distances)
#             elif self.rightT == 3:
#                 min_distances[pred!=labels] = 0 #没有预测对的，先把方向预测对
#                 # print((min_distances>0).sum(),self.T_k)
#                 if self.confidence_thresh>0.0001:
#                     T = 1+ (self.T_k*min_distances)
#                 else:
#                     T = 0.001+ torch.exp(self.T_k*min_distances)
#             else:
#                 T = 1 + (self.T_max - 1) * torch.exp(-self.T_k* min_distances)

#         if return_distance:
#             return min_distances
#         else:
#             return T




class Feather_statistic_append(nn.Module):
    def __init__(self, confidence_thresh=0.0,T_max = 100,T_k = 10,rightT=1,args=None):
        super(Feather_statistic_append, self).__init__()
        self.Queue_mus_list = []
        self.Queue_sigmas_list = []
        self.confidence_thresh = confidence_thresh
        self.T_max = T_max
        self.T_k = T_k
        self.rightT = rightT
        self.args = args


    def append(self,mus,sigmas,Qneue_lenghth = 50000):
        if self.args.Qneue_lenghth > 0 :
            Qneue_lenghth = self.args.Qneue_lenghth
        if self.args.data == 'imagenet':
            Qneue_lenghth = 880880
        mus = mus.view(-1).tolist()
        sigmas = sigmas.view(-1).tolist()

        self.Queue_mus_list.extend(mus)
        if len(self.Queue_mus_list)> Qneue_lenghth:
            self.Queue_mus_list = self.Queue_mus_list[-Qneue_lenghth:]

        self.Queue_sigmas_list.extend(sigmas)
        if len(self.Queue_sigmas_list) > Qneue_lenghth:
            self.Queue_sigmas_list = self.Queue_sigmas_list[-Qneue_lenghth:]

    def forward(self, features, labels,pred,confidence=None,modify_T = False,append=True,return_distance = False):
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        T =  torch.ones_like(labels)
        min_distances = torch.zeros_like(labels)
        correctly_classified = torch.zeros_like(labels)
        if confidence!=None:
            if pred==None:
                pred = labels
            correctly_classified[(labels==pred)&(confidence>self.confidence_thresh)] = 1
            
            means = torch.mean(features, dim=[1])
            stds = torch.std(features, dim=[1])

        if modify_T == True and correctly_classified.sum()!=correctly_classified.numel() and len(self.Queue_mus_list)!=0:

            means = means.unsqueeze(1)  # [num_samples_A, 1]
            stds = stds.unsqueeze(1)    # [num_samples_A, 1]
            meansB_expanded = torch.tensor(self.Queue_mus_list).cuda().unsqueeze(0)
            stdsB_expanded = torch.tensor(self.Queue_sigmas_list).cuda().unsqueeze(0)

            wasserstein_distances = torch.sqrt((means - meansB_expanded) ** 2 + (stds - stdsB_expanded) ** 2)

            min_distances, min_indices = torch.min(wasserstein_distances, dim=1)

            T = torch.exp(-self.T_k* min_distances)

            if self.args.T_scale == 1:
                T = 1 - (min_distances - min_distances.min())/(min_distances.max()-min_distances.min()+1e-8)

        if append==True:
            self.append(means[correctly_classified==1],stds[correctly_classified==1])
        if return_distance:
            return min_distances
        else:
            return T



# class Feather_statistic_append_quantile(nn.Module):
#     def __init__(self, confidence_thresh=0.0,T_max = 5,Quan_number = 4,T_k = 5,rightT=0,args=None):
#         super(Feather_statistic_append_quantile, self).__init__()
#         self.Queue_quan_value_list = []
#         self.confidence_thresh = args.confidence_thresh
#         print('Quan_number',args.Quan_number)
#         self.Quan_number = args.Quan_number
#         self.quantiles =  [i / args.Quan_number for i in range(1, args.Quan_number)]
#         self.T_k = args.T_k
#         self.args = args
#         self.weights = self.generate_weights()

#     def generate_weights(self):
#         import numpy as np
#         # 生成一个线性间隔的数组，中间值为0，两端为负和正
#         if self.Quan_number==0:
#             Quan_number = 2
#         else:
#             Quan_number = self.Quan_number
#         x = np.linspace(-1, 1, Quan_number-1)
#         # 使用高斯公式计算权重，中间最高，两端逐渐减小
#         weights = np.exp(-np.square(x) / 0.5)
#         # 归一化权重使得总和为1
#         weights /= weights.sum()
#         weights = torch.tensor(weights).unsqueeze(0).unsqueeze(0).cuda()
#         return weights
#     def compute_quantiles(self,tensor):
#         return torch.quantile(tensor, torch.tensor(self.quantiles).cuda(), dim=1).permute(1,0)
#     def compute_correlation(self,tensorA, tensorB):
#         # 计算均值
#         mean_A = tensorA.mean(dim=1, keepdim=True)  # torch.Size([256, 1])
#         mean_B = tensorB.mean(dim=1, keepdim=True)  # torch.Size([4000, 1])
        
#         # 计算标准差
#         std_A = tensorA.std(dim=1, keepdim=True)  # torch.Size([256, 1])
#         std_B = tensorB.std(dim=1, keepdim=True)  # torch.Size([4000, 1])
        
#         # 标准化张量
#         normalized_A = (tensorA - mean_A) / std_A  # torch.Size([256, 3])
#         normalized_B = (tensorB - mean_B) / std_B  # torch.Size([4000, 3])
        
#         # 计算相关性
#         correlation = torch.matmul(normalized_A, normalized_B.t()) / (tensorA.shape[1] - 1)  # torch.Size([256, 4000])
        
#         return correlation

#     def append(self,quan_value,Qneue_lenghth = 50000):

#         quan_value = quan_value.tolist()

#         # print(len(self.Queue_quan_value_list))
#         self.Queue_quan_value_list.extend(quan_value)
#         if len(self.Queue_quan_value_list)> Qneue_lenghth:
#             self.Queue_quan_value_list = self.Queue_quan_value_list[-Qneue_lenghth:]


#     def forward(self, features, labels,pred,confidence=None,modify_T = False,append=True,return_distance = False):
#         """
#         Args:
#             features: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         features = features.view(features.shape[0],-1)
#         T =  torch.ones_like(labels)
#         min_distances = torch.zeros_like(labels)
#         correctly_classified = torch.zeros_like(labels)
#         if confidence!=None:
#             if pred==None:
#                 pred = labels
#             correctly_classified[(labels==pred)&(confidence>self.confidence_thresh)] = 1

#             if self.Quan_number == 0:
#                 quan_value = features
#             else:
#                 quan_value = self.compute_quantiles(features)
#             # import pdb
#             # pdb.set_trace()
# # 
#         if modify_T == True and correctly_classified.sum()!=correctly_classified.numel() and len(self.Queue_quan_value_list)!=0:
            
#             if self.args.distance_measure == 'L2_ori':
#                 tensorB = torch.tensor(self.Queue_quan_value_list).cuda()
#                 quantile_vectors_A = quan_value.unsqueeze(1)
#                 quantile_vectors_B = tensorB.unsqueeze(0)      
#                 diffs = quantile_vectors_A - quantile_vectors_B
#                 distances = torch.norm(diffs, dim=2)
#                 min_distances, min_indices = torch.min(distances, dim=1)
#             elif self.args.distance_measure == 'L2':
#                 # tensorB_list = torch.tensor(self.Queue_quan_value_list).cuda()
#                 len_quan_list = len(self.Queue_quan_value_list)
#                 batch_size = 2000  # 根据显存大小调整批次大小
#                 num_batches = (len_quan_list + batch_size - 1) // batch_size
                
#                 min_distances = torch.full((quan_value.shape[0],), float('inf')).cuda()
#                 min_indices = torch.zeros(quan_value.shape[0], dtype=torch.long).cuda()
                
#                 for i in range(num_batches):
#                     start_idx = i * batch_size
#                     end_idx = min((i + 1) * batch_size, len_quan_list)
                    
#                     # tensorB = tensorB_list[start_idx:end_idx,:]
#                     tensorB = torch.tensor(self.Queue_quan_value_list[start_idx:end_idx]).cuda()

#                     quantile_vectors_A = quan_value.unsqueeze(1)
#                     quantile_vectors_B = tensorB.unsqueeze(0)
                    
#                     # if self.args.distance_measure == 'L2':
#                     diffs = quantile_vectors_A - quantile_vectors_B
#                     distances = torch.norm(diffs, dim=2)
#                     batch_min_distances, batch_min_indices = torch.min(distances, dim=1)
                    
#                     mask = batch_min_distances < min_distances
#                     min_distances[mask] = batch_min_distances[mask]
#                     min_indices[mask] = batch_min_indices[mask] + start_idx
#             else:
#                 min_distances = self.compute_distances(quan_value,self.Queue_quan_value_list)
            
#             if self.args.distance_measure == 'weighted_sum':
#                 T = torch.exp(-self.T_k* min_distances*min_distances)
#             else:
#                 T = torch.exp(-self.T_k* min_distances)

#             if self.args.T_scale == 1:
#                 T = 1 - (min_distances - min_distances.min())/(min_distances.max()-min_distances.min()+1e-8)
    

#                 # print(T.max(),T.min(),min_distances.max(),min_distances.min(),len(self.Queue_quan_value_list),'******************')
#                                         #预测正确的典型样本，调整方向，Beta大，靠近1
#                                         #预测正确的非典型样本，Beta接近0
#         if append==True:

#             self.append(quan_value[correctly_classified==1])
#             # self.append(quan_value)

#         if return_distance:
#             return min_distances
#         else:
#             return T
#     def compute_distances(self,tensorA, quan_list):
#         tensorB = torch.tensor(quan_list).cuda()
#         quantile_vectors_A = tensorA.unsqueeze(1)
#         quantile_vectors_B = tensorB.unsqueeze(0)
#         # 计算欧氏距离
#         # 使用unsqueeze扩展维度以进行广播
#         if self.args.distance_measure == 'L2':
#             diffs = quantile_vectors_A - quantile_vectors_B
#             distances = torch.norm(diffs, dim=2)
#         elif self.args.distance_measure == 'wasserstein':
#             diffs_abs = torch.abs(quantile_vectors_A - quantile_vectors_B)

#             # 计算Wasserstein距离（即所有分位点差的绝对值的和）
#             distances = diffs_abs.sum(dim=2)
#         elif self.args.distance_measure == 'weighted_sum':

#             # 分位点差异的加权和
#             # diffs = quantile_vectors_A - quantile_vectors_B
#             diffs = torch.abs(quantile_vectors_A - quantile_vectors_B)
#             # diffs[diffs<0]=0
#             # diffs = diffs-diffs.min()
#             weighted_diffs = diffs * self.weights
#             distances = weighted_diffs.sum(dim=2)
#         elif self.args.distance_measure == 'weighted_mean':
#             # 分位点差异的加权平均值

#             diffs = quantile_vectors_A - quantile_vectors_B
#             # diffs = torch.abs(quantile_vectors_A - quantile_vectors_B)
#             diffs = diffs-diffs.min()
#             # diffs[diffs<0]=0
#             weighted_diffs = diffs * self.weights
#             weighted_diffs[weighted_diffs<0]=0

#             distances = weighted_diffs.mean(dim=2)
#         elif self.args.distance_measure == 'combined':
#             # 基于分位点的距离度量的组合
#             diffs = quantile_vectors_A - quantile_vectors_B
#             # diffs = torch.abs(quantile_vectors_A - quantile_vectors_B)
#             diffs[diffs<0]=0
#             weighted_diffs = diffs * self.weights
#             weighted_sum = weighted_diffs.sum(dim=2)
#             weighted_mean = weighted_diffs.mean(dim=2)
#             max_diff = torch.abs(diffs).max(dim=2)[0]
#             distances = weighted_sum + weighted_mean + max_diff
#         elif self.args.distance_measure == 'correlation':
#             # 分位点差异的相关性
#             # import pdb
#             # pdb.set_trace()
#             distances = self.compute_correlation(tensorA,tensorB)
#         else:
#             print('wrong, no such distance')
#         min_distances, _ = torch.min(distances, dim=1)
#         return min_distances      

# class Feather_statistic_knn(nn.Module):
#     def __init__(self, confidence_thresh=0.0,T_max = 10,T_k = 5,rightT=0):
#         super(Feather_statistic_knn, self).__init__()
#         self.confidence_thresh = confidence_thresh
#         self.T_max = T_max
#         self.T_k = T_k
#         self.rightT = rightT


#     def append(self,mus,sigmas,Qneue_lenghth = 50000):
#         mus = mus.tolist()
#         sigmas = sigmas.tolist()

#         self.Queue_mus_list.extend(mus)
#         if len(self.Queue_mus_list)> Qneue_lenghth:
#             self.Queue_mus_list = self.Queue_mus_list[-Qneue_lenghth:]

#         self.Queue_sigmas_list.extend(sigmas)
#         if len(self.Queue_sigmas_list) > Qneue_lenghth:
#             self.Queue_sigmas_list = self.Queue_sigmas_list[-Qneue_lenghth:]

#     def forward(self, features, labels,pred,confidence=None,modify_T = False,append=True,return_distance = False):
#         """
#         Args:
#             features: feature matrix with shape (batch_size, feat_dim).
#             labels: ground truth labels with shape (batch_size).
#         """
#         T =  torch.ones_like(labels)
#         features = features.view(features.shape[0],-1)
#         correctly_classified = torch.zeros_like(labels)
#         if confidence!=None:
#             if pred==None:
#                 pred = labels
#             correctly_classified[(labels==pred)&confidence>self.confidence_thresh] = 1
#             if append==True:
#                 self.append(features[correctly_classified==1,:])
#         if modify_T == True and correctly_classified.sum()!=correctly_classified.numel() and len(self.Queue_mus_list)!=0:
            
#             means = means.unsqueeze(1)  # [num_samples_A, 1]
#             stds = stds.unsqueeze(1)    # [num_samples_A, 1]
#             meansB_expanded = torch.tensor(self.Queue_mus_list).cuda().unsqueeze(0)
#             stdsB_expanded = torch.tensor(self.Queue_sigmas_list).cuda().unsqueeze(0)

#             # 计算Wasserstein距离矩阵
#             wasserstein_distances = torch.sqrt((means - meansB_expanded) ** 2 + (stds - stdsB_expanded) ** 2)

#             # 找到每个样本对应的最小Wasserstein距离
#             min_distances, min_indices = torch.min(wasserstein_distances, dim=1)
#             # import pdb
#             # pdb.set_trace()
#             if self.rightT == 1:
#                 T = 1 + (self.T_max - 1) * torch.exp(self.T_k*min_distances)
#             elif self.rightT == 2:
#                 T = 0.001+ torch.exp(self.T_k*min_distances)
#             elif self.rightT == 3:
#                 wasserstein_distances[correctly_classified==0] = 0 #没有预测对的，先把方向预测对
#                 T = 0.001+ torch.exp(self.T_k*min_distances)
#             else:
#                 T = 1 + (self.T_max - 1) * torch.exp(-self.T_k* min_distances)
#         if return_distance:
#             return min_distances
#         else:
#             return T