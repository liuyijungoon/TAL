import PIL
from PIL import Image
import torch  # 导入torch库
import numpy as np
import torchvision.transforms as trn
import torchvision
import torchvision as tv
import os
from torch import FloatTensor, div
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# from datasets import load_dataset
import pickle, torch
from torch.utils.data import Subset
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy





def DALI_dataloader(args, dataset='', preprocess='preprocess2', mode='train', debug=0):

    imagenet_root = args.imagenet_root

    if dataset == 'imagenet' or dataset == 'tiny-imagenet':
        if mode == 'train':
            traindir = os.path.join(imagenet_root, 'train')
            pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, 
                                   device_id=args.local_rank, data_dir=traindir, 
                                   crop=224, world_size=args.world_size, local_rank=args.local_rank)
            pipe.build()
            train_loader = DALIClassificationIterator(pipe, size=int(1281167 / args.world_size), 
                                                      auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)
            num_classes = 1000
            # 计算每个 epoch 的 step 数量
            train_size_per_gpu = 1281167 / args.world_size
            train_steps_per_epoch = (train_size_per_gpu + args.batch_size - 1) // args.batch_size
            return train_loader, num_classes, train_steps_per_epoch
        elif mode == 'test':
            valdir = os.path.join(imagenet_root, 'val')
            # world_size = 1
            world_size = args.world_size

            batch_size = args.batch_size*args.world_size
            # local_rank = 0
            local_rank = args.local_rank
            pipe = HybridValPipe(batch_size=batch_size, num_threads=args.workers, 
                                   device_id=local_rank, data_dir=valdir, 
                                   crop=224, world_size=world_size, local_rank=local_rank)
            pipe.build()
            full_val_loader = DALIClassificationIterator(pipe, size=int(50000  / world_size), 
                                                    auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)
            num_classes = 1000
            # 计算每个 epoch 的 step 数量
            test_samples_to_run = 50000
            val_loader = ShuffledLimitedDALIIterator(full_val_loader, test_samples_to_run, seed=12345)
            val_size_per_gpu = test_samples_to_run / world_size
            val_steps_per_epoch = (val_size_per_gpu +batch_size - 1) // batch_size
            return val_loader, num_classes, val_steps_per_epoch

class ShuffledLimitedDALIIterator:
    def __init__(self, dali_iterator, max_samples, seed=None):
        self.dali_iterator = dali_iterator
        self.max_samples = max_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.shuffled_indices = None
        self.current_index = 0
        self.epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.shuffled_indices is None or self.current_index >= len(self.shuffled_indices):
            self._shuffle_and_limit()
            self.current_index = 0
            self.epoch += 1

        if self.current_index >= self.max_samples:
            raise StopIteration

        dali_index = self.shuffled_indices[self.current_index]
        self.dali_iterator.index = dali_index
        batch = next(self.dali_iterator)
        self.current_index += batch[0]['data'].shape[0]

        return batch

    def _shuffle_and_limit(self):
        full_size = self.dali_iterator._size
        self.shuffled_indices = self.rng.permutation(full_size)[:self.max_samples]

    def reset(self):
        self.dali_iterator.reset()
        self.shuffled_indices = None
        self.current_index = 0

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=False)

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]
        
class CustomImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # 获取图像文件路径和标签
        image_path = self.hf_dataset[idx]['image']
        label = self.hf_dataset[idx]['label']

        # 读取图像
        image = Image.open(image_path).convert('RGB')

        # 如果有转换操作，应用它们
        if self.transform:
            image = self.transform(image)

        return image, label
def load_clean_data(csv):
    dataset = load_dataset("csv", data_files=csv)
    img_size = 384
    transformations = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandAugment(num_ops=2,magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
    ])
    trainset = CustomImageDataset(dataset['train'], transform=transformations)
    return trainset



def Dataset(args,dataset = '',preprocess = 'preprocess1',mode = 'train',debug=0):
    imagenet_root = args.imagenet_root
    if preprocess == 'preprocess':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif preprocess == 'preprocess1':
        mean = (0.492, 0.482, 0.446)
        std = (0.247, 0.244, 0.262)

        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                        trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.RandomCrop(32, padding=4),trn.ToTensor(), trn.Normalize(mean, std)])
    elif preprocess == 'preprocess2':
        # import pdb
        # pdb.set_trace()
        if dataset == 'cifar100' or dataset == 'cifar100_svhn':
            mean = [0.507, 0.487, 0.441]
            stdv = [0.267, 0.256, 0.276]
        elif dataset == 'cifar10' or dataset == 'cifar10_svhn':
            mean = [0.491, 0.482, 0.447]
            stdv = [0.247, 0.243, 0.262]
        elif dataset=='imagenet' or dataset == 'imagenet_svhn':
            mean=[0.485, 0.456, 0.406]
            stdv=[0.229, 0.224, 0.225]
 
        # augmentation
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

 


    if mode == 'train':
        if dataset == 'cifar10':
            train_data = torchvision.datasets.CIFAR10(root=args.data_path+"CIFAR10",    
                                                        download=True,  
                                                        train=True,    
                                                        transform=train_transform)
            num_classes = 10
        elif dataset == 'cifar100':
            # import pdb
            # pdb.set_trace()
            train_data = torchvision.datasets.CIFAR100(root=args.data_path+"CIFAR100",  
                                                        download=True,  
                                                        train=True,     
                                                        transform=train_transform
                                                    )
            num_classes = 100
        elif dataset == 'imagenet':
            # traindir = os.path.join('/data0/ImageNet', 'train')
            traindir = os.path.join(imagenet_root, 'train')
            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            train_data = torchvision.datasets.ImageFolder(
                                                traindir,
                                                transforms.Compose([
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    normalize,
                                                ]))
            if debug == 1:
        
                import numpy as np
                subset_indices = np.random.choice(len(train_data), 1000, replace=False)
                # 用随机选择的索引创建数据集的子集
                train_data = Subset(train_data, subset_indices)
        
            num_classes = 1000
        else:
            print('no such dataset')
        return  train_data,num_classes
            
            
    if mode == 'test':
        if dataset == 'cifar10':
            
            test_data = torchvision.datasets.CIFAR10(root=args.data_path+"CIFAR10",    
                                                    download=True, 
                                                    train=False,    
                                                    transform=test_transform
                                                    )
            num_classes = 10
        elif dataset == 'cifar100':

            test_data = torchvision.datasets.CIFAR100(root=args.data_path+"CIFAR100",  
                                                        download=True,  
                                                            train=False,   
                                                            transform=test_transform)
            num_classes = 100
        elif 'svhn' in dataset:
            test_data = torchvision.datasets.SVHN(
                                                    root=args.data_path+"SVHN",    
                                                    download=True,  
                                                    split='test',    
                                                    transform=test_transform
                                                    )
            num_classes = 10
        elif dataset == 'omniglot':
            test_data = torchvision.datasets.Omniglot(root=args.data_path+"Omniglot", 
                                                    background=False, 
                                                    download=True,
                                                    transform=test_transform)
            num_classes = 10
        elif dataset == 'lsun':
            test_data = torchvision.datasets.LSUN(root=args.data_path+"LSUN", 
                                                classes = 'test', 
                                                transform=test_transform)
            num_classes = 10
        elif dataset == 'places365':
            test_data = torchvision.datasets.Places365(root=args.data_path+"Places365",
                                                    split= 'val', 
                                                    small=True, 
                                                    download=True, 
                                                    transform=test_transform)
            num_classes = 10
        elif dataset == 'imagenet':
            # valdir = os.path.join('/data0/ImageNet', 'val')
            # valdir = os.path.join('/home/a100test2/zxy/data/imagenet-1k/imagenet/', 'val')
            valdir = os.path.join(imagenet_root, 'val')

            
            
            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
            test_data = torchvision.datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
            if debug == 1:
        
                import numpy as np
                subset_indices = np.random.choice(len(test_data), 1000, replace=False)
                # 用随机选择的索引创建数据集的子集
                test_data = Subset(test_data, subset_indices)
            num_classes = 1000
        else:
            print('*********************************no such dataset')
        return test_data,num_classes
            
def Loss(loss='cross_entropy'):
    if loss == 'teacher_student':
        from util import symmetric_mse_loss
        loss_func = symmetric_mse_loss
    elif loss == 'RKdAngle':
        from util import RKdAngle
        loss_func = RKdAngle()
    elif loss == 'RkdDistance':
        from util import RkdDistance
        loss_func = RkdDistance()
    elif loss == 'RKD_entropy':
        from util import RKD_entropy
        loss_func = RKD_entropy()

        
    elif loss == 'cross_entropy':
        loss_func = torch.nn.CrossEntropyLoss()  # 'none', 'mean' or 'sum'
    elif loss == 'cross_entropy1':
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')  # 'none', 'mean' or 'sum'
    elif loss == 'LogitNorm':
        from util import LogitNorm
        loss_func = LogitNorm()
        
    elif loss == 'LogitNorm_1':
        from util import LogitNorm_1
        loss_func = LogitNorm_1()
        
    elif loss == 'LogitNorm_2':
        from util import LogitNorm_2
        loss_func = LogitNorm_2()
        
    elif loss == 'LogitNorm_3':
        from util import LogitNorm_3
        loss_func = LogitNorm_3()
        
    elif loss == 'LogitNorm_adaptiveT':
        from util import LogitNorm_adaptiveT
        loss_func = LogitNorm_adaptiveT()
        
    elif loss == 'LogitNorm_adaptiveT1':
        from util import LogitNorm_adaptiveT1
        loss_func = LogitNorm_adaptiveT1()
        
    elif loss == 'FD_yijun':
        from util import FD_yijun
        loss_func = FD_yijun()
        
    elif loss == 'FD_yijun1':
        from util import FD_yijun1
        loss_func = FD_yijun1()
        
    elif loss == 'FD_yijun1_1':
        from util import FD_yijun1_1
        loss_func = FD_yijun1_1()
        
    elif loss == 'FD_yijun1_2':
        from util import FD_yijun1_2
        loss_func = FD_yijun1_2()
        
    elif loss == 'FD_yijun1_adaptiveT':
        from util import FD_yijun1_adaptiveT
        loss_func = FD_yijun1_adaptiveT()
        
    elif loss == 'FD_yijun1_adaptiveT1':
        from util import FD_yijun1_adaptiveT1
        loss_func = FD_yijun1_adaptiveT1()
        
    elif loss == 'FD_yijun1_3':
        from util import FD_yijun1_3
        loss_func = FD_yijun1_3()
        
    elif loss == 'FD_yijun2':
        from util import FD_yijun2
        loss_func = FD_yijun2()
        
    elif loss == 'FD_yijun3':
        from util import FD_yijun3
        loss_func = FD_yijun3()
        
    elif loss == 'FD_yijun6':
        from util import FD_yijun6
        loss_func = FD_yijun6()
        
    elif loss == 'FD_yijun5':
        from util import FD_yijun5
        loss_func = FD_yijun5()
    
    
    elif loss == 'FD_yijun1_1_inverse':
        from util import FD_yijun1_1_inverse
        loss_func = FD_yijun1_1_inverse()
    elif loss == 'SCELoss':
        from util import SCELoss
        loss_func = SCELoss()
    else:
        print("no such loss")
    return loss_func

def Model(model_name = 'wrn',dataset= 'cifar10'):
    if dataset== 'cifar10':
        num_class = 10
    if dataset== 'cifar100':
        num_class = 100
    if dataset== 'imagenet':
        num_class = 1000

    if model_name == 'ori_model':
        from model import Net_CIFAR10_90_no_drop
        model = Net_CIFAR10_90_no_drop()
    # elif model_name == 'wrn' and dataset=='cifar10':
    #     from model import WideResNet
    #     model = WideResNet(40, 10, 2, dropRate=0.3)
    elif model_name == 'resnet18':
        import torchvision.models as models
        resnet18 = models.resnet18(pretrained=False)
        resnet18.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet18.maxpool = torch.nn.Identity()
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_class) 
        model = resnet18
    elif model_name == 'resnet50':
        import torchvision.models as models
        resnet18 = models.resnet50(pretrained=False)
        resnet18.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet18.maxpool = torch.nn.Identity()
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, num_class) 
        model = resnet18
    elif model_name == 'swin' and dataset=='tiny-imagenet':
        model = create_model('swin_large_patch4_window12_384', pretrained=True, drop_path_rate=0.1)
    elif model_name == 'dense':
        from models import densenet_BC
        model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                        growth_rate=12, reduction=0.5,
                                        bottleneck=True, dropRate=0.0).cuda()
    elif model_name == 'wrn':
        from models import wrn
        model = wrn.WideResNet(28, num_class, 10).cuda()
    else:
        print(model_name,dataset,'not paired')
    return model