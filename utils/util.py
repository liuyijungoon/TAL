import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
import pdb
import numpy as np
import math
pi = math.pi
from collections import OrderedDict

def load_weights_from_file(
    model, weights_path, dev="cuda", keep_last_layer=True, 
):
    """Load parameters from a path of a file of a state_dict."""

    # special case for google BiTS model


    state_dict = torch.load(weights_path, map_location=dev)

    # special case for pretrained torchvision model
    # they fudged their original state dict and didn't change it
  

    new_state_dict = OrderedDict()

    # data parallel trained models have module in state dict
    # prune this out of keys
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    # load params
    state_dict = new_state_dict
    if not keep_last_layer:

        # filter out final linear layer weights
        state_dict = {
            key: params for (key, params) in state_dict.items()
            if "classifier" not in key and "fc" not in key
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        print("loading weights")
        model.load_state_dict(state_dict, strict=True)




    
class LogitNorm(nn.Module):

    def __init__(self,t=100.0):
        super(LogitNorm, self).__init__()
        self.t = t
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y):
        
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) 
        

        return F.cross_entropy(logit_norm* self.t, y)
    

class LogitNorm1(nn.Module):

    def __init__(self,epi = 0):
        super(LogitNorm1, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.epi = epi

    def forward(self, x, y,T=None):
        # pdb.set_trace()
        if T==None:
            T = torch.ones_like(y)
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) 
        
        return F.cross_entropy(logit_norm* T.unsqueeze(1), y)

class LogitNorm2(nn.Module):

    def __init__(self):
        super(LogitNorm2, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y,T=None):
        # pdb.set_trace()
        if T==None:
            T = torch.ones_like(y)
        # norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        # logit_norm = torch.div(x, norms) 
        
        return F.cross_entropy(x* T.unsqueeze(1), y)
    

class LogitNorm_2(nn.Module):

    def __init__(self,t=1.0):
        super(LogitNorm_2, self).__init__()
        self.t = t
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y):
        # pdb.set_trace()
        
    
        
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) 
        

        return F.cross_entropy(logit_norm*25, y)
class LogitNorm_3(nn.Module):

    def __init__(self,t=1.0):
        super(LogitNorm_3, self).__init__()
        self.t = t
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y):
        # import pdb
        # pdb.set_trace()
    
        
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms)*100
        

        return F.cross_entropy(logit_norm, y)
    


    
class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
#########################################看这儿，2023.4.8

class cosDistanceLoss2(nn.Module):

    def __init__(self,t=1.0):
        super(cosDistanceLoss2, self).__init__()
        self.t = t
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y):
        # pdb.set_trace()
        
        b = x.shape[0]
        int_batch = y.long()
        one_hot_batch = -torch.zeros(b, 10).cuda()
        one_hot_batch.scatter_(1, int_batch.unsqueeze(1), 1)
        
        
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) 
        
        # norms_y = torch.norm(one_hot_batch, p=2, dim=1, keepdim=True) + 1e-7
        # y = torch.div(one_hot_batch, norms_y) / self.t
        
        x = self.softmax(logit_norm)
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) 
        # pdb.set_trace()
        # cos = 1-torch.sum(torch.mul(x,one_hot_batch),dim=1)
        cos = torch.sum(torch.mul(logit_norm,one_hot_batch),dim=1)
        
        
        # return torch.mean(cos)
        return cos
class cosDistanceLoss1(nn.Module): ###cos

    def __init__(self,t=1.0):
        super(cosDistanceLoss1, self).__init__()
        self.t = t
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, y):
        # pdb.set_trace()
        
        b = x.shape[0]
        int_batch = y.long()
        one_hot_batch = torch.zeros(b, 10).cuda()
        one_hot_batch.scatter_(1, int_batch.unsqueeze(1), 1)
        
        
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) 
        

        cos = torch.sum(torch.mul(logit_norm,one_hot_batch),dim=1)
        
        
        # return torch.mean(cos)
        return cos,norms
    

    
class LogitNormLoss(nn.Module):

    def __init__(self, t=1):
        super(LogitNormLoss, self).__init__()
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)

def fprX(y_pred,y_true, tpr_threshold = 0.95):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # # Find the threshold corresponding to 80% TPR
    # tpr_threshold = 0.8
    idx = next(i for i, tpr in enumerate(tpr) if tpr >= tpr_threshold)
    threshold_at_X_tpr = thresholds[idx]

    # Calculate the FPR at 80% TPR
    tn = sum((y_true == 0) & (y_pred < threshold_at_X_tpr))
    fp = sum((y_true == 1) & (y_pred < threshold_at_X_tpr))
    fpr_at_X_tpr = fp / (tn + fp)
    
    return fpr_at_X_tpr





    

    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
    
def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """

    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    input1 = torch.nn.Softmax(dim=1)(input1)
    input2 = torch.nn.Softmax(dim=1)(input2)

    return torch.sum((input1 - input2)**2) / num_classes

def get_entropy( x):
    # x : N x C
    x = F.normalize(x, p=2, dim=-1) # N x C
    # 将归一化后的x转换为概率分布
    probabilities = F.softmax(x, dim=-1)
    # 计算熵
    # 使用clamp函数防止对数函数中的数值稳定性问题
    entropy = -probabilities * torch.log(probabilities.clamp(min=1e-9))
    # 对每个样本的熵求和
    entropy = entropy.sum(dim=-1)
    return entropy
def symmetric_mse_loss_entropy(input1, input2,reduce = None):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """

    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    input1 = torch.nn.Softmax(dim=1)(input1)
    input2 = torch.nn.Softmax(dim=1)(input2)
    if reduce==None:
        return torch.sum((input1 - input2)**2) / num_classes
    else:
        return (input1 - input2)**2


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups