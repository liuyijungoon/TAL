import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from torchvision import datasets, transforms
from scipy.spatial import distance
import torch.distributed as dist

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
import time

# 导入其他需要的模块
from model import resnet, resnet18, densenet_BC, vgg, mobilenet, efficientnet, wrn, convmixer
from utils import data as dataset, crl_utils, metrics, utils
# import train_fmfp, train_src, train_fmfpnorm, train_EFC, train_fmfpEFC
import wandb
from trainer import train_fmfp
from trainer import train_fmfpnorm
from trainer import train_TAL
from trainer import train_openmix
from trainer import train_base

from utils.data_utils import *
import torchvision.models as models
from torch.optim.swa_utils import AveragedModel, SWALR
from utils.sam import SAM
from torch.nn.modules.batchnorm import _BatchNorm


# ... (保留其他导入和函数定义)
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
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
def randomly_zero_weights(model, layers_to_modify, zero_probability):
    """
    Randomly set a portion of weights to zero in specified layers.
    
    :param model: The PyTorch model
    :param layers_to_modify: List of layer names to modify
    :param zero_probability: Probability of setting a weight to zero
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_modify):
            mask = torch.rand_like(param) > zero_probability
            param.data *= mask
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
        return model
parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--epochs', default=90, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--plot', default=20, type=int, help='')
parser.add_argument('--runs', default=0, type=int, help='')
parser.add_argument('--classnumber', default=1000, type=int, help='class number for the dataset')
parser.add_argument('--data', default='imagenet', type=str, help='Dataset name to use [cifar10, cifar100, imagenet,tiny-imagenet]')
parser.add_argument('--model', default='resnet50', type=str, help='Models name to use [res110, dense, wrn, cmixer, efficientnet, mobilenet, vgg]')
parser.add_argument('--method', default='baseline', type=str, help='[sam, swa, fmfp]')
parser.add_argument('--data_path', default='../../data/', type=str, help='Dataset directory')
parser.add_argument('--save_path', default='./output/', type=str, help='Savefiles directory')
parser.add_argument('--rank_weight', default=1.0, type=float, help='Rank loss weight')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--scheduler', default='steplr', type=str, help='')
parser.add_argument('--scheduler_Tmax', default=0, type=int, help='')
parser.add_argument('--T_scale', default=1, type=int, help='')
parser.add_argument('--correct_first', default=3, type=int, help='')
parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.5)
parser.add_argument("--avg_type", help="pya estimation types", type=str, default='batch')
parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
parser.add_argument('--DALI', type=int,default=1)


parser.add_argument("--lambda_dis_align",  help="lambda_dis in Eq.2", type=float, default=2.0)
parser.add_argument('--append_afterSAM', default=1, type=int, help='')
parser.add_argument('--use_wandb', type=int,default=1)
parser.add_argument('--changeT', type=int,default=1)

parser.add_argument('--imagenet_root', default='/dataset/sharedir/research/ImageNet/', type=str, help='')


parser.add_argument('--Quan_number', type=int,default=0)
parser.add_argument('--distance_measure', type=str,default='L2')


parser.add_argument('--workers', default='8', type=int, help='dataloader')
parser.add_argument('--debug', default='0', type=int, help='debug')

parser.add_argument('--no_progress', default='0', type=int, help='no_progress')

parser.add_argument('--modify_T_begin_epoch', type=int, default=10)
parser.add_argument('--beta_consistency', type=float,default=1)
parser.add_argument('--lambda_value', type=float,default=1)

parser.add_argument('--teacher', type=int,default=0)

parser.add_argument('--confidence_thresh', type=float,default=0.0)
parser.add_argument('--weight_decay', type=float,default=0.0001)
parser.add_argument('--swa_lr', type=float,default=0.05)
parser.add_argument('-aux_set', type=str, default='RandomImages', help='RandomImages')



parser.add_argument('--T_max', type=int,default=100)
parser.add_argument('--T_min', type=float,default=10)
parser.add_argument('-aux_size', type=int, default=-1, help='using all RandomImages data')


parser.add_argument('--testlast', type=int,default=0)
parser.add_argument('--pca', type=int,default=0)

parser.add_argument('--append_right', type=int,default=1)
parser.add_argument('--autocast', type=int,default=0)
parser.add_argument('--resume', type=str,default='None')
parser.add_argument('--pretrain', type=int,default=0)
parser.add_argument('--Qneue_lenghth', type=int,default=0)
parser.add_argument('--direct', type=int,default=0)
parser.add_argument('--try_data_range', type=int,default=0)
parser.add_argument('--partially_zero', type=int,default=0)

parser.add_argument('--T_k', type=float,default=10.0)
parser.add_argument('--epi', type=float,default=0)
parser.add_argument('--base_lr', type=float,default=0.1)
parser.add_argument('--local-rank', type=int, default=-1, metavar='N',
                    help='Local process rank.')

parser.add_argument('--rightT', type=int,default=0)
parser.add_argument('--lambda_o', type=float, default=1, help='[0.1, 0.5, 1.0, 1.5, 2] dnl loss weight')
parser.add_argument("--tau", help="loss tau", type=float, default=0.1)
parser.add_argument("--feature_statistic", type=int, default=1)
parser.add_argument('--print-freq', '-p', default=200, type=int, metavar='N', help='print frequency (default: 10)')


def main():
    args = parser.parse_args()
    args.world_size = torch.cuda.device_count()
    
    if args.local_rank is not None:
        main_worker(args.local_rank, args)
    else:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))

def main_worker(gpu, args):
    args.gpu = gpu
    
    if args.local_rank is not None:
        args.rank = args.local_rank
        args.gpu = args.local_rank
    else:
        args.rank = gpu
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    args.batch_size = args.batch_size // args.world_size
    torch.cuda.set_device(args.gpu)
    if args.append_right == 0:
        from utils.utils import Feather_statistic 
    elif args.append_right == 2:
        from utils.utils import Feather_statistic_append_quantile as Feather_statistic 
    else:
        from utils.utils import Feather_statistic_append as Feather_statistic 
        
    if args.scheduler_Tmax == 0:
        args.scheduler_Tmax = args.epochs
    if args.use_wandb and dist.get_rank() == 0:
        import wandb
        my_wandb = wandb.init(
        name=args.method,
        project="After_Nips",
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

    # Ensure that cuDNN is used in a deterministic manner
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.data == 'cifar100':
        num_class = 100
    elif args.data == 'imagenet':
        num_class = 1000
    elif args.data == 'tiny-imagenet':
        num_class = 200
    else:
        num_class = 10
    if args.method == 'openmix' or args.method == 'openmix_LogitNorm3':
        num_class = num_class + 1
    args.num_classes = num_class
    model_dict = {"num_classes": num_class}
    calculate_metrics_yijun = metrics.calculate_metrics_yijun(args)
    

    save_path = args.save_path + args.data + '_' + args.model + '_' + args.method + '_' + args.scheduler+ '_correct_first' + str(args.correct_first)+ '_' + str(args.T_max)+ '_' + str(args.T_k)+ '_run' + str(args.runs)


    ranking_criterion = None
    
    os.makedirs(save_path,exist_ok=True)
    teacher_model = None
    num_class = args.num_classes

    correctness_history = None
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    if args.testlast == 1:
        test_begin_epoch = args.epochs-1
    else:
        test_begin_epoch = 0
    if args.pretrain==1:
        pretrained = True
    else:
        pretrained = False
    base_lr = 0.1  # Initial learning rate
    if args.pretrain == 1:
        if args.data == 'imagenet':
            base_lr = 0.001
        else:
            base_lr = 0.01

    lr_strat = [int(80/200*args.epochs),int(140/200*args.epochs)]
    lr_factor = 0.1  # Learning rate decrease factor
    custom_weight_decay = args.weight_decay # Weight Decay
    custom_momentum = 0.9  # Momentum





    if args.data == 'imagenet' or args.data == 'tiny-imagenet':
        from utils.setting import DALI_dataloader
        train_loader,num_class,_ = DALI_dataloader(args,args.data,'preprocess2',mode = 'train',debug=args.debug)
    else:
        train_loader, valid_loader, test_loader, test_onehot, test_label = dataset.get_loader(args.data, args.data_path, args.batch_size, args)
        train_sampler = DistributedSampler(train_loader.dataset)
        train_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)



    model = get_model(args)
    if args.resume!='None':
        from utils.util import load_weights_from_file
        load_weights_from_file(model, args.resume)


    model.cuda(args.gpu)
    model = DDP(model, device_ids=[args.gpu])

    Feather_statistic1 = Feather_statistic(confidence_thresh=args.confidence_thresh,T_max = args.T_max,T_k = args.T_k,rightT=args.rightT,args=args)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum,weight_decay=custom_weight_decay)
    if args.scheduler == 'steplr':
        if args.data == 'imagenet':
                scheduler =StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        print('no such scheduler')
    if 'LogitNorm' in args.method or args.method == 'TAL':
        from utils.util import LogitNorm2,LogitNorm1,LogitNorm
        cls_criterion1 = LogitNorm1(args.epi)
        cls_criterion2 = LogitNorm2() 
        if args.data == 'imagenet' and args.method=='LogitNorm':
            cls_criterion = LogitNorm(1000)
        else:
            cls_criterion = LogitNorm(args.T_max)

    else:
        cls_criterion = nn.CrossEntropyLoss().cuda()
    import pdb
    
    if args.resume == 'None':
        begin_epoch = 0
    else:
        import re
        match = re.search(r'/epoch(\d+)/', args.resume)
        print(args.resume)
        if match:
            epoch_number = match.group(1)
            print(f"Epoch number: {epoch_number}")
        else:
            print("Epoch number not found")
        begin_epoch = int(epoch_number)+1


    print(args.resume,begin_epoch)
    for epoch in range(begin_epoch, args.epochs):
        if args.DALI != 1:
            train_sampler.set_epoch(epoch)

        if args.method == 'baseline' or args.method == 'LogitNorm':
            train_base.train(train_loader, model, cls_criterion, ranking_criterion, optimizer, epoch, correctness_history, train_logger, args)
            scheduler.step()

            if epoch >= test_begin_epoch:
                if args.method == 'LogitNorm':
                    result = calculate_metrics_yijun.test_epoch(model.module, epoch, uncertainty_metric='msp')
                   
                    if args.use_wandb and dist.get_rank() == 0:
                        wandb_record(my_wandb, result[0], 'OldFD', epoch, result[2], uncertainty_metric='cos')
                        wandb_record(my_wandb, result[1], 'NewFD', epoch, result[2], uncertainty_metric='cos')
                else:
                    result = calculate_metrics_yijun.test_epoch(model.module, epoch, uncertainty_metric='msp')
                    if args.use_wandb and dist.get_rank() == 0:
                        wandb_record(my_wandb, result[0], 'OldFD', epoch, result[2], uncertainty_metric='msp')
                        wandb_record(my_wandb, result[1], 'NewFD', epoch, result[2], uncertainty_metric='msp')
                if dist.get_rank() == 0:
                    save_path_pth = os.path.join(save_path, f'epoch{epoch}')
                    print(save_path_pth)
                    os.makedirs(save_path_pth, exist_ok=True)
                    torch.save(model.module.state_dict(), os.path.join(save_path_pth, 'model.pth'))

        elif  args.method == 'TAL':
            
            if  'TAL' == args.method:

                train_TAL.train(train_loader, model, cls_criterion1, optimizer, epoch, correctness_history, train_logger, args,Feather_statistic1,criterion1 = cls_criterion2)
                         
            scheduler.step()
            # if epoch>args.epochs-5: 
            result = calculate_metrics_yijun.test_epoch(model,epoch,uncertainty_metric='cos')
            if epoch>=test_begin_epoch and dist.get_rank() == 0:
                if args.use_wandb:
                    wandb_record(my_wandb,result[0],'OldFD',epoch,result[2],uncertainty_metric='cos')
                    wandb_record(my_wandb,result[1],'NewFD',epoch,result[2],uncertainty_metric='cos')
                save_path_pth = os.path.join(save_path, 'epoch'+str(epoch))
                os.makedirs(save_path_pth,exist_ok=True)
                print(save_path_pth)
                torch.save(model.state_dict(),os.path.join(save_path_pth, 'model.pth'))


    dist.destroy_process_group()

if __name__ == "__main__":
    main()