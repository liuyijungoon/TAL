
import utils.crl_utils
from utils import utils
import torch.nn as nn
import time
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

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
class RKD_entropy(nn.Module):
    def forward(self, student, teacher,reduction='mean'):
        # N x C
        # N x N x C

        with torch.no_grad():
            teacher = F.normalize(teacher, p=2, dim=-1) # n, c
            t_angle = teacher @ teacher.permute(1, 0)     
            t_soft = F.softmax(t_angle, dim=-1)
          
            entropy = -1 * t_soft * torch.log(t_soft)
            entropy = entropy.sum(-1) /teacher.shape[0]

        student = F.normalize(student, p=2, dim=-1) # n, c
        s_angle = student @ student.permute(1, 0)  
        
        loss = torch.abs(s_angle-t_angle)*entropy
        if reduction=='mean':
            return loss.mean()
        else:
            return torch.mean(loss, dim=1)
def update_ema_variables(args,model, ema_model, start, global_step,sharpness_score=None,strict_alpha = False,total_step=None):
    # Use the true average until the exponential average is more correct
    
    alpha = min(1 - 1 / (global_step + 1), start)
    k = args.update_teacher_k
    if alpha == start:
        x = global_step/(total_step-np.ceil(1/(1-start)))
        start = 0.999
        if args.update_teacher_strategy == 'linear': #y=0.999+(1−0.999)x
            alpha = start + (1-start)*x
        elif args.update_teacher_strategy == 'exp':  #y=0.999+(1−0.999)(1−e^(−10x)
            alpha = start + (1-start)*(1-np.exp(-k*x))
        elif args.update_teacher_strategy == 'power':  #y_log =0.999 + (1 - 0.999) * x**10
            alpha =start + (1 - start) * x**k

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss
kdloss = KDLoss(2.0)


def classAug(args, x, y, base_numclass, alpha=10.0, mix_times=2):
    batch_size = x.size()[0]
    mix_data = []
    mix_target = []
    new_classnum = args.newclassnum
    for _ in range(mix_times):
        index = torch.randperm(batch_size).cuda()
        for i in range(batch_size):
            if y[i] != y[index][i]:
                new_label = generate_label(y[i].item(), y[index][i].item(), base_numclass, new_classnum)
                lam = np.random.beta(alpha, alpha)
                if lam < 0.4 or lam > 0.6:
                    lam = 0.5
                mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                mix_target.append(new_label)

    new_target = torch.Tensor(mix_target)
    y = torch.cat((y, new_target.cuda().long()), 0)
    for item in mix_data:
        x = torch.cat((x, item.unsqueeze(0)), 0)
    return x, y


def generate_label(y_a, y_b, base_numclass=10, new_classnum=1):
    y_a, y_b = y_a, y_b
    assert y_a != y_b
    if y_a > y_b:
        tmp = y_a
        y_a = y_b
        y_b = tmp
    label_index = ((2 * base_numclass - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
    return (label_index % new_classnum) + base_numclass


def mixup_data(x, y, alpha=0.3, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def RegMixup_data(x, y, alpha=10, use_cuda=True):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
    
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 100 * sigmoid_rampup(epoch, 10)

def OE_mixup(x_in, x_out, alpha=10.0):
    if x_in.size()[0] != x_out.size()[0]:
        length = min(x_in.size()[0], x_out.size()[0])
        x_in = x_in[:length]
        x_out = x_out[:length]
    lam = np.random.beta(alpha, alpha)
    x_oe = lam * x_in + (1 - lam) * x_out
    return x_oe, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(loader, loader_out, model, criterion, criterion_ranking, optimizer, epoch, history, logger, args,teacher_model = None,scaler=None):
    from utils.utils import AverageMeter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    cls_losses = AverageMeter()
    ranking_losses = AverageMeter()
    end = time.time()
    model.train()

    if args.method == 'openmix':
        for index, (in_set, out_set) in enumerate(tqdm(zip(loader, loader_out),desc=f"Epoch {epoch} training",disable=args.no_progress)):
            in_data, out_data, target = in_set[0].cuda(), out_set[0].cuda(), in_set[1].cuda()
            in_oe, lam = OE_mixup(in_data, out_data)
            inputs = torch.cat([in_data, in_oe], dim=0)


        if scaler==None:           
            logits = model(inputs)
            target_oe = torch.LongTensor(in_oe.shape[0]).random_(args.num_classes, args.num_classes + 1).cuda()
            loss_in = F.cross_entropy(logits[:in_data.shape[0]], target)
            loss_oe = lam * F.cross_entropy(logits[in_data.shape[0]:], target[:target_oe.shape[0]]) + (1 - lam) * F.cross_entropy(logits[in_data.shape[0]:], target_oe)
            loss = loss_in + args.lambda_o * loss_oe
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:    

            optimizer.zero_grad()  
            with autocast():     
                logits = model(inputs)
                target_oe = torch.LongTensor(in_oe.shape[0]).random_(args.num_classes, args.num_classes + 1).cuda()
                loss_in = F.cross_entropy(logits[:in_data.shape[0]], target)
                loss_oe = lam * F.cross_entropy(logits[in_data.shape[0]:], target[:target_oe.shape[0]]) + (1 - lam) * F.cross_entropy(logits[in_data.shape[0]:], target_oe)
                loss = loss_in + args.lambda_o * loss_oe

            scaler.scale(loss).backward()
            scaler.step(optimizer)    
            scaler.update()  




            prec, correct = utils.accuracy(logits[:in_data.shape[0]], target)
            total_losses.update(loss.item(), logits[:in_data.shape[0]].size(0))
            top1.update(prec.item(), logits[:in_data.shape[0]].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.write([epoch, total_losses.avg, top1.avg])
        
    elif args.method == 'openmix_LogitNorm3':
        teacher_model.train()
        RKD_entropy_criterion = RKD_entropy()
        i = 0
        total_step = len(loader)*args.batch_size
        for in_set, out_set in zip(loader, loader_out):
            globel_step = epoch*args.batch_size+i
            i = i+1
            in_data, out_data, target = in_set[0].cuda(), out_set[0].cuda(), in_set[1].cuda()
            in_oe, lam = OE_mixup(in_data, out_data)
            inputs = torch.cat([in_data, in_oe], dim=0)
            logits = model(inputs)
            
            
            teacher_outputs = teacher_model(inputs)
                
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * symmetric_mse_loss(teacher_outputs.detach(),logits) / inputs.shape[0]*args.num_classes/100
            
            
            
            
            RKD_entropy_angle_loss = RKD_entropy_criterion(logits, teacher_outputs.detach(),reduction='none')
            RKD_loss =  2*RKD_entropy_angle_loss.mean()
                    
                    
            
            target_oe = torch.LongTensor(in_oe.shape[0]).random_(args.num_classes, args.num_classes + 1).cuda()
            
            
            
            loss_in = F.cross_entropy(logits[:in_data.shape[0]], target)
            loss_oe = lam * F.cross_entropy(logits[in_data.shape[0]:], target[:target_oe.shape[0]]) \
                      + (1 - lam) * F.cross_entropy(logits[in_data.shape[0]:], target_oe)
                      
                      
                      
            loss = loss_in + args.lambda_o * loss_oe+RKD_loss+consistency_loss
            
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            update_ema_variables(args,model,teacher_model,0.999,globel_step,total_step=total_step)
            
            
            prec, correct = utils.accuracy(logits[:in_data.shape[0]], target)
            total_losses.update(loss.item(), logits[:in_data.shape[0]].size(0))
            top1.update(prec.item(), logits[:in_data.shape[0]].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.write([epoch, total_losses.avg, top1.avg])
        
        
   