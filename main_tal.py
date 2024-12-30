
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from scipy.spatial import distance
from scipy.stats import chi2
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import normalize
import argparse
import os
import csv
import math
import pandas as pd
import numpy as np
import random
import resource
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
import time

from model import resnet
from model import resnet18
from model import densenet_BC
from model import vgg
from model import mobilenet
from model import efficientnet
from model import wrn
from model import convmixer
from utils import data as dataset
from utils import crl_utils
from utils import metrics
from utils import utils

from trainer import train_fmfp
from trainer import train_fmfpnorm
from trainer import train_TAL
from trainer import train_openmix
from utils.data_utils import *

import torchvision.models as models
from torch.optim.lr_scheduler import StepLR



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
from trainer import train_base
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.sam import SAM
from torch.nn.modules.batchnorm import _BatchNorm

def update_bn(swa_model, loader):
    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, _BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = swa_model.training
    swa_model.train()
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.cuda()
        swa_model(batch)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]

    swa_model.train(was_training)
def csv_writter(path, dic, start):
    if os.path.isdir(path) == False: os.makedirs(path)
    os.chdir(path)
    # Write dic
    if start == 1:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 1:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])


class Counter(dict):
    def __missing__(self, key):
        return None

def wandb_record(wandb,metric_ones,flagg,epoch,acc,uncertainty_metric= 'msp'):
    wandb.log(
                {
                    'epoch':epoch,
                    flagg+'_fpr95':metric_ones[5], 
                    flagg+'_aupr_s':metric_ones[3], 
                    flagg+'_aupr_e':metric_ones[4], 
                    flagg+'_tnr95':metric_ones[6], 
                    flagg+'_aurc':metric_ones[0], 
                    flagg+'_eaurc':metric_ones[1], 
                    flagg+'_auroc':metric_ones[2], 
                    flagg+'_acc':acc, 
                    'uncertainty_metric':uncertainty_metric,

                }
            )
def get_model(args):
    if args.method == 'openmix':
        num_class = num_class + 1
    model_dict = {
        "num_classes": num_class,
    }
    teacher_model = None
    num_class = args.num_classes
    if args.pretrain==1:
        pretrained = True
    else:
        pretrained = False
    model_dict = {"num_classes": num_class}
    if args.model == 'resnet18':
        model = resnet18.ResNet18(**model_dict).cuda()
    elif args.model == 'deit':
        import timm

        if args.data == 'imagenet' or args.data == 'tiny-imagenet':
            model = timm.create_model('deit_small_patch16_224', pretrained=pretrained)
            if pretrained==True:
                num_features = model.head.in_features
                model.head = nn.Linear(num_features, num_class)
        else:
            model = timm.create_model('deit_small_patch16_224', pretrained=pretrained, num_classes=num_class, img_size=32, patch_size=4)
        model = model.cuda()
    elif args.model == 'resnet50':
        import torchvision.models as models
        model = models.__dict__[args.model](pretrained = pretrained)
        if args.teacher==1:
            teacher_model = models.__dict__[args.model]()
            num_features = teacher_model.fc.in_features  # 获取最后一个全连接层的输入特征数
            teacher_model.fc = nn.Linear(num_features, num_class)  # 创建新的全连接层

        model =model.cuda()

    elif args.model == 'res110':
        model = resnet.resnet110(**model_dict).cuda()
        if args.teacher==1:
            teacher_model = resnet.resnet110(num_class)

    elif args.model == 'dense':
        model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                        growth_rate=12, reduction=0.5,
                                        bottleneck=True, dropRate=0.0).cuda()
    elif args.model == 'vgg':
        model = vgg.vgg16(**model_dict).cuda()
    elif args.model == 'wrn':
        model = wrn.WideResNet(28, num_class, 10).cuda()
    elif args.model == 'efficientnet':
        model = efficientnet.efficientnet(**model_dict).cuda()
    elif args.model == 'mobilenet':
        model = mobilenet.mobilenet(**model_dict).cuda()
    elif args.model == "cmixer":
        model = convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_class).cuda()
    if args.teacher==1:
        return model,teacher_model
    else:
        return model,None
parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--epochs', default=90, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--plot', default=20, type=int, help='')
parser.add_argument('--runs', default=0, type=int, help='')
parser.add_argument('--classnumber', default=1000, type=int, help='class number for the dataset')
parser.add_argument('--data', default='imagenet', type=str, help='Dataset name to use [cifar10, cifar100, imagenet]')
parser.add_argument('--model', default='resnet50', type=str, help='Models name to use [res110, dense, wrn, cmixer, efficientnet, mobilenet, vgg]')
parser.add_argument('--method', default='TAL', type=str, help='[sam, swa, fmfp]')
parser.add_argument('--data_path', default='../../data/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./output/', type=str, help='Savefiles directory')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--scheduler', default='steplr', type=str, help='')
parser.add_argument('--scheduler_Tmax', default=0, type=int, help='')
parser.add_argument('--T_scale', default=1, type=int, help='')
parser.add_argument('--correct_first', default=3, type=int, help='')
parser.add_argument('--DALI', type=int,default=0)


parser.add_argument('--append_afterSAM', default=1, type=int, help='')
parser.add_argument('--use_wandb', type=int,default=1)
parser.add_argument('--changeT', type=int,default=2)

parser.add_argument('--imagenet_root', default='../../imagenet/', type=str, help='')

parser.add_argument('--Quan_number', type=int,default=0)
parser.add_argument('--distance_measure', type=str,default='L2')


parser.add_argument('--workers', default='32', type=int, help='dataloader')
parser.add_argument('--debug', default='0', type=int, help='debug')
parser.add_argument('--no_progress', default='0', type=int, help='no_progress')
parser.add_argument('--modify_T_begin_epoch', type=int, default=5)
parser.add_argument('--beta_consistency', type=float,default=1)
parser.add_argument('--lambda_value', type=float,default=1)

parser.add_argument('--teacher', type=int,default=0)

parser.add_argument('--confidence_thresh', type=float,default=0.0)
parser.add_argument('--weight_decay', type=float,default=0.0005)
parser.add_argument('--swa_lr', type=float,default=0.05)
parser.add_argument('-aux_set', type=str, default='RandomImages', help='RandomImages')
parser.add_argument('--exp_mode1', type=str, default='typicalness', help='RandomImages')




parser.add_argument('--T_max', type=int,default=100)
parser.add_argument('--T_min', type=float,default=10)
parser.add_argument('-aux_size', type=int, default=-1, help='using all RandomImages data')


parser.add_argument('--testlast', type=int,default=1)
parser.add_argument('--pca', type=int,default=0)

parser.add_argument('--append_right', type=int,default=1)
parser.add_argument('--autocast', type=int,default=0)
parser.add_argument('--resume', type=int,default=0)
parser.add_argument('--pretrain', type=int,default=0)
parser.add_argument('--Qneue_lenghth', type=int,default=0)


parser.add_argument('--T_k', type=float,default=10.0)
parser.add_argument('--epi', type=float,default=0)
parser.add_argument('--rightT', type=int,default=1)
parser.add_argument('--lambda_o', type=float, default=1, help='[0.1, 0.5, 1.0, 1.5, 2] dnl loss weight')

parser.add_argument("--feature_statistic", type=int, default=1)
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()



def main(args):
    run_time = 1
    from utils.utils import Feather_statistic_append as Feather_statistic 
        
    if args.scheduler_Tmax == 0:
        args.scheduler_Tmax = args.epochs
        
    if args.use_wandb:
        import wandb
        my_wandb = wandb.init(
        name=args.method,
        project="Nips",
        entity="yijun",
        mode="online",
        save_code=True,
        
        config=args,
        notes='compare'
    )
    args.modify_T_begin_epoch = int(10/200*args.epochs)


    if args.runs == 0:
        seed = 1234
    elif args.runs == 1:
        seed = 4321
    elif args.runs == 2:
        seed = 5656

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    calculate_metrics_yijun = metrics.calculate_metrics_yijun(args)
    
    save_path = args.save_path + args.data + '_' + args.model + '_' + args.method + '_' + args.scheduler+ '_correct_first' + str(args.correct_first)+ '_' + str(args.T_max)+ '_' + str(args.T_k)+ '_run' + str(args.runs)



    os.makedirs(save_path,exist_ok=True)
    if args.data == 'imagenet':
        from utils.setting import Dataset
        train_dataset,num_class = Dataset(args,args.data,'preprocess2',mode = 'train',debug=args.debug)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    else:
        train_loader, valid_loader, test_loader, \
        test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)

    if args.data == 'cifar100':
        num_class = 100
    elif args.data == 'imagenet':
        num_class = 1000
    elif args.data == 'cifar10':
        num_class = 10
    args.num_classes = num_class

    

    for r in range(run_time):
        print(100 * '#')
        print(args.runs)
        model,teacher_model = get_model(args)
        # if args.model == 'resnet18':
        #     model = resnet18.ResNet18(**model_dict).cuda()
        # elif args.model == 'deit':
        #     import timm
        #     if args.pretrain == 1:
        #         pretrained = True
        #     else:
        #         pretrained = False
        #     model = timm.create_model('deit_small_patch16_224', pretrained=pretrained, num_classes=num_class, img_size=32, patch_size=4)
        #     model = model.cuda()
        # elif args.model == 'resnet50':
        #     import torchvision.models as models
        #     model = models.__dict__[args.model]()
            
        #     if args.teacher==1:
        #         teacher_model = models.__dict__[args.model]()
        #         num_features = teacher_model.fc.in_features  # 获取最后一个全连接层的输入特征数
        #         teacher_model.fc = nn.Linear(num_features, num_class)  # 创建新的全连接层

        #     num_features = model.fc.in_features  # 获取最后一个全连接层的输入特征数
        #     model.fc = nn.Linear(num_features, num_class)  # 创建新的全连接层
        #     model =model.cuda()

        # elif args.model == 'res110':
        #     model = resnet.resnet110(**model_dict).cuda()
        #     if args.teacher==1:
        #         teacher_model = resnet.resnet110(**model_dict)

        # elif args.model == 'dense':
        #     model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
        #                                   growth_rate=12, reduction=0.5,
        #                                   bottleneck=True, dropRate=0.0).cuda()
        # elif args.model == 'vgg':
        #     model = vgg.vgg16(**model_dict).cuda()
        # elif args.model == 'wrn':
        #     model = wrn.WideResNet(28, num_class, 10).cuda()
        # elif args.model == 'efficientnet':
        #     model = efficientnet.efficientnet(**model_dict).cuda()
        # elif args.model == 'mobilenet':
        #     model = mobilenet.mobilenet(**model_dict).cuda()
        # elif args.model == "cmixer":
        #     model = convmixer.ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=num_class).cuda()

        if 'LogitNorm' in args.method or 'TAL' in args.method:
            from utils.util import LogitNorm2,LogitNorm1,LogitNorm
            cls_criterion1 = LogitNorm1(args.epi)
            cls_criterion2 = LogitNorm2() 
            
            cls_criterion = LogitNorm(args.T_max)

        else:
            cls_criterion = nn.CrossEntropyLoss().cuda()

        if args.resume==1:
            pathh = os.path.join(save_path, 'epoch'+str(199))
            pathh =  os.path.join(pathh, 'model.pth')
            checkpoint = torch.load(pathh)
            model.load_state_dict(checkpoint)

        if args.teacher==1:
            for param in teacher_model.parameters():
                param.requires_grad = False 
            teacher_model.cuda()

        if args.autocast==1:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
        else:
            scaler = None
        # make logger
        train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
        result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

        correctness_history = crl_utils.History(len(train_loader.dataset))
        ranking_criterion = nn.MarginRankingLoss(margin=0.0).cuda()

        base_lr = 0.1  # Initial learning rate
        if args.pretrain == 1:
            base_lr = 0.01
        lr_strat = [80, 130, 170]
        lr_factor = 0.1  # Learning rate decrease factor
        custom_weight_decay = args.weight_decay # Weight Decay
        custom_momentum = 0.9  # Momentum
        if args.method == 'openmix':
            ood_data, _ = build_dataset(args, args.aux_set, "train", data_num=args.aux_size,
                                    origin_dataset=args.data)
            train_loader_out = torch.utils.data.DataLoader(ood_data,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        if 'fmfp' in args.method:
            if args.model == "convmixer":
                base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=custom_weight_decay)
                optimizer = SAM(model.parameters(), base_optimizer)
            else:
                base_optimizer = torch.optim.SGD
                optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum,
                                weight_decay=custom_weight_decay)
            
            if args.scheduler == 'steplr':
                scheduler =StepLR(optimizer, step_size=30, gamma=0.1)
            elif args.scheduler == 'cos':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_Tmax)
            elif args.scheduler == 'multisteplr':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80,140], gamma=0.1)
            else:
                print('no such scheduler')
            swa_model = AveragedModel(model)
            swa_start = int(120/200*args.epochs)
            swa_scheduler = SWALR(optimizer, swa_lr=0.05)


            
        elif args.method == 'baseline' or args.method == 'LogitNorm' or args.method == 'openmix':
            Feather_statistic1 = Feather_statistic(confidence_thresh=args.confidence_thresh,T_max = args.T_max,T_k = args.T_k,rightT=args.rightT,args=args)
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,weight_decay=custom_weight_decay)
            if args.scheduler == 'steplr':
                scheduler =StepLR(optimizer, step_size=30, gamma=0.1)
            elif args.scheduler == 'cos':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_Tmax)
            elif args.scheduler == 'multisteplr':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80,140], gamma=0.1)
            else:
                print('no such scheduler')
        elif 'TAL' in args.method :
            Feather_statistic1 = Feather_statistic(confidence_thresh=args.confidence_thresh,T_max = args.T_max,T_k = args.T_k,rightT=args.rightT,args=args)

            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,
                                        weight_decay=custom_weight_decay)
                
            if args.scheduler == 'steplr':
                scheduler =StepLR(optimizer, step_size=30, gamma=0.1)
            elif args.scheduler == 'cos':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_Tmax)
            elif args.scheduler == 'multisteplr':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80,140], gamma=0.1)
            else:
                print('no such scheduler')
            swa_model = AveragedModel(model)
            swa_start = int(120/200*args.epochs)
            swa_scheduler = SWALR(optimizer, swa_lr=0.05)

        elif 'LogitNorm' in args.method:

            Feather_statistic1 = Feather_statistic(confidence_thresh=args.confidence_thresh,T_max = args.T_max,T_k = args.T_k,rightT=args.rightT,args=args)
            
            base_optimizer = torch.optim.SGD
            optimizer = SAM(model.parameters(), base_optimizer, lr=base_lr, momentum=custom_momentum,
                                weight_decay=custom_weight_decay)
                
            if args.scheduler == 'steplr':
                scheduler =StepLR(optimizer, step_size=30, gamma=0.1)
            elif args.scheduler == 'cos':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_Tmax)
            elif args.scheduler == 'multisteplr':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80,140], gamma=0.1)
            else:
                print('no such scheduler')
            swa_model = AveragedModel(model)
            swa_start = int(120/200*args.epochs)
            swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        # make logger
        train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
        result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

        correctness_history = crl_utils.History(len(train_loader.dataset))
        if args.testlast == 1:
            test_begin_epoch = args.epochs-1
        else:
            test_begin_epoch = 0
        # start Train

        for epoch in range(0, args.epochs):
            if args.method == 'baseline' or args.method == 'LogitNorm':
                if args.resume == 1:
                    result = calculate_metrics_yijun.test_epoch(model,199,uncertainty_metric='cos')
                    result = calculate_metrics_yijun.test_epoch(model,199,uncertainty_metric='msp')
                    result = calculate_metrics_yijun.test_epoch(model,199,uncertainty_metric='maxLogit')
                    result = calculate_metrics_yijun.test_epoch(model,199,uncertainty_metric='energy')
                    break
                else:
                    train_base.train(train_loader,model,cls_criterion,ranking_criterion,optimizer,epoch,correctness_history,train_logger,args,scaler=scaler) 
                    scheduler.step()

                    if epoch>=test_begin_epoch: 
                        if args.method == 'LogitNorm':
                            result = calculate_metrics_yijun.test_epoch(model,epoch,uncertainty_metric='cos')
                            if args.use_wandb:
                                wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='cos')
                                wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='cos')
                        else:
                            result = calculate_metrics_yijun.test_epoch(model,epoch,uncertainty_metric='msp')

                            if args.use_wandb:
                                wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='msp')
                                wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='msp')


                        save_path_pth = os.path.join(save_path, 'epoch'+str(epoch))
                        os.makedirs(save_path_pth,exist_ok=True)
                        torch.save(model.state_dict(),
                                os.path.join(save_path_pth, 'model.pth'))
            
            elif args.method == 'TAL':
                if  'TAL' == args.method:
                    train_TAL.train(train_loader, model, cls_criterion1, optimizer, epoch, correctness_history, train_logger, args,Feather_statistic1,criterion1 = cls_criterion2,scaler=scaler)               
                scheduler.step()
                if epoch>=test_begin_epoch: 
                    result = calculate_metrics_yijun.test_epoch(model,epoch,uncertainty_metric='cos')

                    if args.use_wandb:
                        wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='cos')
                        wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='cos')
                    save_path_pth = os.path.join(save_path, 'epoch'+str(epoch))
                    os.makedirs(save_path_pth,exist_ok=True)
                    torch.save(model.state_dict(),os.path.join(save_path_pth, 'model.pth'))
            
            elif args.method == 'fmfp' and args.confidence_thresh==0.0:
                train_fmfp.train(train_loader,
                            model,
                            cls_criterion,
                            ranking_criterion,
                            optimizer,
                            epoch,
                            correctness_history,
                            train_logger,
                            args)
                if args.method == 'swa' or args.method == 'fmfp':
                    if scheduler != None:
                        if epoch > swa_start:
                            swa_model.update_parameters(model)
                            swa_scheduler.step()
                        else:
                            scheduler.step()
                else:
                    if scheduler != None:
                        scheduler.step()
   
                if epoch>=test_begin_epoch:                                                                
                    print('updata bn')
                    update_bn(swa_model, train_loader)
                    result = calculate_metrics_yijun.test_epoch(swa_model,epoch,uncertainty_metric='msp')
                    if args.use_wandb:
                        wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='msp')
                        wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='msp')
                    save_path_pth = os.path.join(save_path, 'epoch'+str(epoch))
                    os.makedirs(save_path_pth,exist_ok=True)
                    torch.save(swa_model.state_dict(),
                               os.path.join(save_path_pth, 'model.pth'))
            elif args.method == 'openmix':


                train_openmix.train(train_loader, train_loader_out,model,cls_criterion,ranking_criterion,optimizer,epoch,correctness_history,train_logger,args)
                scheduler.step()
                if epoch>=test_begin_epoch: 
                    result = calculate_metrics_yijun.test_epoch(model,epoch,uncertainty_metric='msp')
                    if args.use_wandb:
                        wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='msp')
                        wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='msp')
                    save_path_pth = os.path.join(save_path, 'epoch'+str(epoch))
                    os.makedirs(save_path_pth,exist_ok=True)
                    torch.save(model.state_dict(),os.path.join(save_path_pth, 'model.pth'))

            elif 'fmfp_TAL' in args.method:
        
                if  args.method == 'fmfp_TAL':
                    train_fmfpnorm.train(train_loader, model, cls_criterion1, optimizer, epoch, correctness_history, train_logger, args,Feather_statistic1,criterion1 = cls_criterion2)  
                if epoch > swa_start:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    scheduler.step()
                if epoch>=test_begin_epoch: 
                                                                            
                    print('updata bn')
                    update_bn(swa_model, train_loader)
        
                    result = calculate_metrics_yijun.test_epoch(swa_model,epoch,uncertainty_metric='cos')
                    if args.use_wandb:
                        wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='cos')
                        wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='cos')
                    save_path_pth = os.path.join(save_path, 'epoch'+str(epoch))
                    os.makedirs(save_path_pth,exist_ok=True)
                    torch.save(swa_model.state_dict(),
                            os.path.join(save_path_pth, 'model.pth'))  
                    torch.save(model.state_dict(),
                            os.path.join(save_path_pth, 'student_model.pth'))  


if __name__ == "__main__":
    main(args)






