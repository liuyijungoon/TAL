import torch.nn.functional as F
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac
tl.set_backend('pytorch')
import numpy as np
from utils import utils
import utils.crl_utils
import time
import torch
from tqdm import tqdm




def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def pca_torch(X, n_components=8):
    # 标准化数据
    X = X.view(X.shape[0],-1)
    X_mean = torch.mean(X, dim=0)
    X = X - X_mean
    
    # 计算协方差矩阵
    cov_matrix = torch.matmul(X.T, X) / (X.shape[0] - 1)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix, UPLO='U')
    
    # 选择最大的 n_components 个特征向量
    principal_components = eigenvectors[:, -n_components:]
    
    # 投影数据到低维空间
    X_reduced = torch.matmul(X, principal_components)
    
    return X_reduced

def update_ema_variables(model, ema_model, start, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), start)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return  sigmoid_rampup(epoch,5)
def train(loader, model, criterion, optimizer, epoch, history, logger, args,Feather_statistic1,criterion1 = None,only_forward=False,scaler=None):
    from utils.utils import AverageMeter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    top1 = AverageMeter()
    cls_losses = AverageMeter()
    ranking_losses = AverageMeter()
    end = time.time()
    
    if only_forward:
        model.eval()  
    else:
        model.train()  
    loss_avg = 0.0
    yes = 0
    no = 0


    consistency_loss = 0
    sharpness_score = 0

    features = None

    if args.feature_statistic:
        def train_hook_fn(module, input, output):
            nonlocal features
            module_name = type(module).__name__
            if isinstance(module, torch.nn.Linear):
                features = input[0].data
                if args.pca==1:
                    # features = pca_torch(features, n_components=args.Quan_number)
                    pool = nn.AdaptiveAvgPool2d((args.Quan_number,1))
                    features = pool(features)
                elif args.pca == 2:
                    features = features.view(features.shape[0],-1)
                    weights, factors = parafac(features, rank=args.Quan_number)
                    features = tl.cp_to_tensor((weights, factors))

        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Linear):
                module.register_forward_hook(train_hook_fn)
                break
    consistency_loss_list = []
    for index,dataa in enumerate(tqdm(loader,desc=f"Epoch {epoch} training",disable=args.no_progress)):
        if args.DALI:
            imgs = dataa[0]["data"]
            labels = dataa[0]["label"].squeeze(-1).long()
        else:
            imgs, labels = dataa[0],dataa[1]
        globel_step = epoch*args.batch_size+index
        imgs = imgs.cuda()
        labels = labels.cuda()  

        if scaler==None:
            outputs = model(imgs)
            if only_forward==True:
                break
            # confidence,pred =torch.nn.Softmax(1)(outputs).max(dim=1)
            confidence, pred = F.normalize(outputs, p=2, dim=1).max(1)
            if args.feature_statistic:
                if args.append_right == 0:
                    if epoch>=args.modify_T_begin_epoch:
                        T = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = True)
                    else:
                        T = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = False)


                    if args.changeT == 0 and epoch>=args.modify_T_begin_epoch:
                        T1 = torch.ones_like(labels)*args.T_max
                        T2 = torch.ones_like(labels)
                    elif args.rightT>1 and args.changeT == 1 and epoch>=args.modify_T_begin_epoch:
                        T1 = T**args.T_max
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
                else:
                    if epoch>=args.modify_T_begin_epoch:
                        Beta = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = True)
                    else:
                        Beta = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = False)
                    if args.changeT==1:
                        T1 =torch.ones_like(labels)*(args.T_min+Beta*(args.T_max-args.T_min))
                    elif args.changeT==2:
                        T1 =torch.ones_like(labels)*(args.T_min+(1-Beta)*(args.T_max-args.T_min))
                    else:
                        T1 = torch.ones_like(labels)*args.T_max
                    if index==1:
                        print(epoch,Beta.max(),Beta.min(),(confidence>args.confidence_thresh).sum())
                    
                    T2 = torch.ones_like(labels)
                    Beta[pred!=labels] = 1 #预测错误的，全调整方向，
                    
                                            #预测正确的典型样本，调整方向，Beta大，靠近1
                                            #预测正确的非典型样本，Beta接近0
                    #2024，5，19
                    
                    if args.correct_first ==2:
                        #预测错误的首先应该致力于预测正确，而LogitNorm在vit网络上提取特征的能力相较于cross entropy损失太多，因此：
                        cross_entropy_loss = (1-Beta)*criterion(outputs, labels,T1) + Beta*(args.lambda_value)*criterion1(outputs, labels,T2)
                    elif args.correct_first ==1:
                        Beta[pred!=labels] = 0 #预测错误的，全用cross entropy预测正确再说，因为cross entropy比logitNorm强
                        if criterion1!= None:
                            cross_entropy_loss = Beta*criterion(outputs, labels,T1) + (1-Beta)*(args.lambda_value)*criterion1(outputs, labels,T2)
                        else:
                            cross_entropy_loss = criterion(outputs, labels,T1)     
                    elif args.correct_first == 3:
                        T1[pred!=labels] = 20
                        if criterion1!= None:
                            cross_entropy_loss = Beta*criterion(outputs, labels,T1) + (1-Beta)*(args.lambda_value)*criterion1(outputs, labels,T2)
                        else:
                            cross_entropy_loss = criterion(outputs, labels,T1)      
                    else:
                        if criterion1!= None:
                            cross_entropy_loss = Beta*criterion(outputs, labels,T1) + (1-Beta)*(args.lambda_value)*criterion1(outputs, labels,T2)
                        else:
                            cross_entropy_loss = criterion(outputs, labels,T1)                              

            
                loss = cross_entropy_loss.mean()
            
            loss_avg = loss_avg * 0.8 + float(loss.item()) * 0.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                if only_forward==True:
                    break
                # confidence,pred =torch.nn.Softmax(1)(outputs).max(dim=1)
                confidence, pred = F.normalize(outputs, p=2, dim=1).max(1)
                if args.feature_statistic:
                    if args.append_right == 0:
                        if epoch>=args.modify_T_begin_epoch:
                            T = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = True)
                        else:
                            T = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = False)


                        if args.changeT == 0 and epoch>=args.modify_T_begin_epoch:
                            T1 = torch.ones_like(labels)*args.T_max
                            T2 = torch.ones_like(labels)
                        elif args.rightT>1 and args.changeT == 1 and epoch>=args.modify_T_begin_epoch:
                            T1 = T**args.T_max
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
                    else:
                        if epoch>=args.modify_T_begin_epoch:
                            Beta = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = True)
                        else:
                            Beta = Feather_statistic1.forward(features, labels,pred=pred,confidence=confidence,modify_T = False)
                        if args.changeT==1:
                            T1 =torch.ones_like(labels)*(args.T_min+Beta*(args.T_max-args.T_min))
                        elif args.changeT==2:
                            T1 =torch.ones_like(labels)*(args.T_min+(1-Beta)*(args.T_max-args.T_min))
                        else:
                            T1 = torch.ones_like(labels)*args.T_max
                        if index==1:
                            print(epoch,Beta.max(),Beta.min(),(confidence>args.confidence_thresh).sum())
                        
                        T2 = torch.ones_like(labels)
                        Beta[pred!=labels] = 1 #预测错误的，全调整方向，
                        
                                                #预测正确的典型样本，调整方向，Beta大，靠近1
                                                #预测正确的非典型样本，Beta接近0
                        if args.correct_first ==2:
                            #预测错误的首先应该致力于预测正确，而LogitNorm在vit网络上提取特征的能力相较于cross entropy损失太多，因此：
                            cross_entropy_loss = (1-Beta)*criterion(outputs, labels,T1) + Beta*(args.lambda_value)*criterion1(outputs, labels,T2)
                        elif args.correct_first ==1:
                            Beta[pred!=labels] = 0
                            Beta[pred == labels] = 1
                            if criterion1!= None:
                                cross_entropy_loss = Beta*criterion(outputs, labels,T1) + (1-Beta)*(args.lambda_value)*criterion1(outputs, labels,T2)
                            else:
                                cross_entropy_loss = criterion(outputs, labels,T1)       
                        else:
                            if criterion1!= None:
                                cross_entropy_loss = Beta*criterion(outputs, labels,T1) + (1-Beta)*(args.lambda_value)*criterion1(outputs, labels,T2)
                            else:
                                cross_entropy_loss = criterion(outputs, labels,T1)        

                
                    loss = cross_entropy_loss.mean()
            
            loss_avg = loss_avg * 0.8 + float(loss.item()) * 0.2

            scaler.scale(loss).backward()
            scaler.step(optimizer)    
            scaler.update()    


        

        prec, correct = utils.accuracy(outputs, labels)

        total_losses.update(loss.item(), imgs.size(0))
        top1.update(prec.item(), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
        #         epoch, i, len(loader), batch_time=batch_time,
        #         data_time=data_time, loss=total_losses, top1=top1))
    if only_forward!=True:
        logger.write([epoch, total_losses.avg, top1.avg])

    if args.feature_statistic:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Linear):
                module._forward_hooks.clear()
                break


        
    