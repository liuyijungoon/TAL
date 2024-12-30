import torch.nn
from utils.setting import Loss
from tqdm import tqdm
import torch.nn.functional as F
from utils import utils
import time
import torch
import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))



def update_ema_variables(model, ema_model, start, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), start)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return  sigmoid_rampup(epoch,5)
def train(loader, model, teacher_model, criterion, optimizer, epoch, history, logger, args,Feather_statistic1,criterion1 = None):
    batch_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    teacher_student_loss = Loss('teacher_student')
    end = time.time()
    
    model.train()  
    if args.teacher:
        teacher_model.train() 
    loss_avg = 0.0


    consistency_loss = 0

    features = None

    if args.feature_statistic:
        def train_hook_fn(module, input, output):
            nonlocal features
            module_name = type(module).__name__
            if isinstance(module, torch.nn.BatchNorm2d):
                features = output.data

        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.BatchNorm2d):
                module.register_forward_hook(train_hook_fn)
                break

    for index,dataa in enumerate(tqdm(loader,desc=f"Epoch {epoch} training",disable=args.no_progress)):
        imgs, labels= dataa[0],dataa[1]
        globel_step = epoch*args.batch_size+index
        imgs = imgs.cuda()
        labels = labels.cuda()
        


        outputs = model(imgs)

        confidence, pred = F.normalize(outputs, p=2, dim=1).max(1)
        if args.feature_statistic:
            if epoch>=args.modify_T_begin_epoch:
                T = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = True)
            else:
                T = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = False)
            if args.changeT == 0 and epoch>=args.modify_T_begin_epoch:
                T1 = torch.ones_like(labels)*args.T_max
                T2 = torch.ones_like(labels)
            elif args.rightT>1 and args.changeT == 1 and epoch>=args.modify_T_begin_epoch:
                T1 = T**args.T_max #这里x20
                T2 = torch.ones_like(labels)
            else:
                T1 = T
                T2 = T

            if criterion1!= None:
                if args.rightT==2:
                    cross_entropy_loss = 1/T*criterion(outputs, labels,T1) + (1-1/T)*criterion1(outputs, labels,T2)
                else:               
                    cross_entropy_loss = (1-1/T)*criterion(outputs, labels,T1) + 1/T*criterion1(outputs, labels,T2)
            else:

                cross_entropy_loss = criterion(outputs, labels,T1)


        if args.teacher:
            with torch.no_grad():
                teacher_outputs = teacher_model(imgs)
            consistency_weight = get_current_consistency_weight(epoch)
            consistency_loss = consistency_weight * teacher_student_loss(teacher_outputs.detach(),outputs) / imgs.shape[0]*args.num_classes #注意，后面这三项是真正的一致性损失，因为util里面的函数没写对
            loss= cross_entropy_loss.mean()+args.beta_consistency*consistency_loss
        else:
            loss = cross_entropy_loss.mean()
        
        loss_avg = loss_avg * 0.8 + float(loss.item()) * 0.2
        optimizer.zero_grad() 
        


                        
        loss.backward()
        

         
        optimizer.first_step(zero_grad=True)


        output_sam = model(imgs)

        if args.feature_statistic:
            if criterion1== None:
                loss_sam = criterion(output_sam, labels,T1).mean()
            else:
                if args.rightT==2:
                    loss_sam =( 1/T*criterion(output_sam, labels,T1) + (1-1/T)*criterion1(output_sam, labels,T2)).mean()
                else:   
                    loss_sam =( (1-1/T)*criterion(output_sam, labels,T1) + 1/T*criterion1(output_sam, labels,T2)).mean()

        else:
            loss_sam = criterion(output_sam, labels).mean()
        


        loss_sam.backward()
        optimizer.second_step(zero_grad=True)


        if args.teacher:
            update_ema_variables(model,teacher_model,0.999,globel_step)

        

        prec, correct = utils.accuracy(teacher_outputs, labels)

        total_losses.update(loss.item(), imgs.size(0))
        top1.update(prec.item(), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.write([epoch, total_losses.avg, top1.avg])

    if args.feature_statistic:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.BatchNorm2d):
                module._forward_hooks.clear()
                break


        
    