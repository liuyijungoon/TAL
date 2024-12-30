
import torch
import torch.nn as nn

 
import pandas as pd
import numpy as np
from utils.eval_utils import *

import torchvision as tv
from torchvision import transforms
from tqdm import tqdm
from argparse import ArgumentParser


# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)
parser.add_argument('--data_path', default='/dataset/vshaozuoyu/liuyijun/data/', type=str, help='Dataset directory')
parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="added to end of filenames to differentiate them if needs be"
)
parser.add_argument('--data', default='imagenet', type=str, help='Dataset name to use [cifar10, cifar100, imagenet,tiny-imagenet]')

parser.add_argument('--ood_data', default='textures', type=str, help='Dataset name to use [cifar10, cifar100, imagenet,tiny-imagenet]')


parser.add_argument('--model', default='deit', type=str, help='Models name to use [res110, dense, wrn, cmixer, efficientnet, mobilenet, vgg]')

parser.add_argument('--imagenet_root', default='/dataset/sharedir/research/ImageNet/', type=str, help='')
parser.add_argument('--debug', default='0', type=int, help='debug')
parser.add_argument('--workers', default='8', type=int, help='dataloader')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--num_classes', default=200, type=int, help='num_classes')
parser.add_argument('--seed', default=1278, type=int, help='seed')
parser.add_argument('--fast', default=1, type=int, help='seed')
parser.add_argument('--sample', default=0, type=int, help='seed')
parser.add_argument('--use_train_as_test', default=0, type=int, help='seed')


parser.add_argument('--weights_path', default='/dataset/vshaozuoyu/liuyijun/code/SIRC/imagenet_resnet50_baseline_100_10.0_run1/epoch89/model.pth', type=str, help='')


args = parser.parse_args()


import random
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure that cuDNN is used in a deterministic manner
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if '/imagenet' in args.weights_path:
    args.data = 'imagenet'
    args.num_classes = 1000

if '/cifar100' in args.weights_path:
    args.data = 'cifar100'
    args.num_classes = 100
if '/tiny-imagenet' in args.weights_path:
    args.data = 'tiny-imagenet'

if 'deit' in args.weights_path:
    args.model = 'deit'

if 'resnet50' in args.weights_path:
    args.model = 'resnet50'

if 'res110' in args.weights_path:
    args.model = 'res110'

if 'wrn' in args.weights_path:
    args.model = 'wrn'

if 'dense' in args.weights_path:
    args.model = 'dense'


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {dev}")


import sys
sys.path.append('/dataset/vshaozuoyu/liuyijun/code/FMFP')
from setting import Dataset
                               
if args.data == 'cifar100':
    args.ood_data = 'svhn'



test_dataset,num_class = Dataset(args,args.data,'preprocess2',mode = 'test',debug=args.debug)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
num_workers=args.workers, pin_memory=True)

print('eval on', args.data)

if args.ood_data == 'textures':

    ood_data,ood_num_classes = Dataset(args,args.data+'_textures','preprocess2',mode = 'test')
    ood_dataloader = torch.utils.data.DataLoader(ood_data, batch_size=256, shuffle=False)


if args.ood_data == 'svhn':

    ood_data,ood_num_classes = Dataset(args,args.data+'_svhn','preprocess2',mode = 'test')
    ood_dataloader = torch.utils.data.DataLoader(ood_data, batch_size=256, shuffle=False)

if args.ood_data == 'wilds':
    from wilds import get_dataset
    from wilds.common.data_loaders import get_eval_loader
    dataset = get_dataset(dataset="camelyon17", download=True)

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # fMoW 推荐的大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 获取测试集
    ood_data = dataset.get_subset("test", transform=transform)

    # 创建数据加载器
    ood_dataloader = get_eval_loader("standard", ood_data, batch_size=args.batch_size, num_workers=4)

# print transforms
# print("="*80)
# print(args.data)
# print(id_data.test_set.transforms)
# print("="*80)
# for data in ood_data:
#     print("="*80)
#     print(data.name)
#     try:
#         print(data.test_set.dataset.transforms)
#     except:
#         print(data.test_set.transforms)
#     print("="*80)


# gmm parameters (means and covariance matrix) ------------------------







if args.fast == 1:
    gmm_params,vim_params,knn_params,train_stats = None,None,None,None
else:
    gmm_path = args.weights_path.replace('model.pth','gmm.pth')
    gmm_params = torch.load(gmm_path)


    vim_path = args.weights_path.replace('model.pth','vim.pth')
    vim_params = torch.load(vim_path)


    knn_path = args.weights_path.replace('model.pth','knn.pth')
    knn_params = torch.load(knn_path)


    stats_path = args.weights_path.replace('model.pth','train_stats.pth')
    train_stats = torch.load(stats_path)
def no_dummy(z):
    def hook(model, input, output):
        z.append(output.squeeze())
        return output
    return hook

def no_dummy_input(z):
    def hook(model, input, output):
        z.append(input[0])
        return input
    return hook


def no_dummy_input1(z):
    def hook(model, input, output):
        z.append(input[0])
        return output  # 返回原始输出，而不是输入
    return hook
# helper functions ------------------------------------------------------------
from sklearn.model_selection import StratifiedShuffleSplit

global early_stop
if args.sample == 1:
    early_stop = None
else:
    # early_stop =30000
    early_stop = len(test_dataset)
    print('len(test_loader)',len(test_loader))

def stratified_sample(vector, n_samples):
    # 确保 n_samples 不大于原始向量长度
    n_samples = min(n_samples, len(vector))
    
    # 如果 n_samples 等于向量长度，直接返回所有索引的随机排列
    if n_samples == len(vector):
        return np.random.permutation(len(vector))
    
    # 创建一个虚拟的索引数组
    indices = np.arange(len(vector))
    
    # 使用 StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=42)
    
    # 获取采样的索引
    for _, sample_indices in sss.split(indices, vector):
        return sample_indices
def get_logits_labels(
    model, loader, 
    dev="cuda", 
    early_stop=None # stop eval early 
):
    """Get the model outputs for a dataloader."""

    model.eval()
    # get ID data
    label_list = []
    logit_list = []
    feature_list = []
    count = 0
    with torch.no_grad():
        for i, dataa in enumerate(tqdm(loader)):
            inputs, labels = dataa[0], dataa[1]
            labels, inputs = labels.to(dev), inputs.to(dev)
            batch_size = inputs.shape[0]
  
            z_b = []
            if args.model =='deit':
                hook_fn = model.head.register_forward_hook(no_dummy_input(z_b))
            elif args.model =='resnet50':
                hook_fn = model.avgpool.register_forward_hook(no_dummy(z_b))
            else:
                hook_fn = model.fc.register_forward_hook(no_dummy_input1(z_b))

       
            logit = model(inputs)


            hook_fn.remove()
            feature = z_b[0]

            if args.model =='deit':
                logit = model.head(feature)
            else:
                logit = model.fc(feature)
   

      
            label_list.append(labels.to("cpu"))

            # in case I fuck up in specifying for the model
            logit_list.append(logit.to("cpu"))
            feature_list.append(feature.to("cpu"))

            count += batch_size
            if (
                early_stop is not None 
                and 
                count >= early_stop
            ):
                break

    logits, labels = torch.cat(logit_list, dim=0), torch.cat(label_list, dim=0)
    features = torch.cat(feature_list, dim=0)
    # clip to exactly match the early stop
    if early_stop is not None:
        logits, labels = logits[:early_stop], labels[:early_stop]

        features = features[:early_stop]

    return logits, labels, features


def evaluate(
    model, test_loader, 
    ood_data=None, dev="cuda",
    shifted=False
):
    """Evaluate the model's topk error rate and ECE."""
    top1 = TopKError(k=1, percent=True)
    top5 = TopKError(k=5, percent=True)
    nll = nn.CrossEntropyLoss()

    logits_dict = {}
    features_dict ={}
    logits, labels, features = get_logits_labels(
        model, test_loader, dev=dev,early_stop=early_stop,
    )



    store = {}
    # store logits for later
    logits_dict[args.data] = logits.to("cpu")

    if features is not None:
        features_dict[args.data] = features.to("cpu")

    results = {}
    results["dataset"] = args.data
    results["top1"] = top1(labels, logits)
    results["top5"] = top5(labels, logits)
    results["nll"] = nll(logits, labels).item() # backwards
    results["acc"] = 1-top1(labels, logits)


    if args.fast == 0:
        metrics = uncertainties(logits, features=features, gmm_params=gmm_params,vim_params=vim_params, knn_params=knn_params,stats=train_stats)
    else:
        metrics = uncertainties(logits, features=None, gmm_params=gmm_params,vim_params=vim_params, knn_params=knn_params,stats=train_stats)




    # record average values 
    res = {f"{args.data} {k}": v.mean().item() for k, v in metrics.items()}
    results.update(res)


    # ID correct vs incorrect
    max_logits, preds = logits.max(dim=-1)
    miscls_labels = (preds != labels)

    store['old_msp'] = metrics['confidence']
    store['old_cos'] = metrics['cos']
    store['old_label'] = miscls_labels
    store['old_features'] = features

    print('ID acc',(preds == labels).sum()/((preds == labels).sum()+(preds != labels).sum()))
    # AUROC
    miscls_res = detect_results(miscls_labels, metrics, mode="ROC")
    print('MisD AUROC',miscls_res)
    miscls_res = {f"{args.data} errROC " + k: v for k, v in miscls_res.items() if k != "mode"}
    results.update(miscls_res)

    
    # FPR@95
    miscls_res = detect_results(miscls_labels, metrics, mode="FPR@95")
    print('MisD FPR95',miscls_res)
    miscls_res = {f"{args.data} errFPR@95 " + k: v for k, v in miscls_res.items() if k != "mode"}
    results.update(miscls_res)

    # AURC
    miscls_res = detect_results(miscls_labels, metrics, mode="AURC")
    print('MisD AURC',miscls_res)

    miscls_res = {f"{args.data} AURC " + k: v for k, v in miscls_res.items() if k != "mode"}
    results.update(miscls_res)


    correct_idx =  torch.zeros_like(labels)
    correct_idx[preds == labels] = 1

    # print(results)


    # OOD data stuff
    if ood_data is not None:
        ood_results = {}
        for data in ood_data:
            print(f"eval on: {args.ood_data}")

            ood_logits, _, ood_features = get_logits_labels(
                model, ood_dataloader, dev=dev,early_stop=early_stop,
            )

                

            # balance the #samples between OOD and ID data
            # unless OOD dataset is smaller than ID, then it will stay smaller
            # this does not happen by default   

            min_len = min(ood_logits.shape[0],logits.shape[0])
            print(min_len)

      
            # 
            if args.sample ==1: 
                import pdb
                pdb.set_trace()
                sampled_indices = stratified_sample(labels, min_len)
                logits1 = logits[sampled_indices]
                print(logits1[-1,1:10])
                # ood_logits = ood_logits[:min_len]
                features1 = features[sampled_indices]
                correct_idx1 = correct_idx[sampled_indices]
                print('sampled acc',correct_idx1.sum()/correct_idx1.shape[0])
            
            else:
                logits1 = logits[:min_len]
                print(logits1[-1,1:10])

                # ood_logits = ood_logits[:min_len]
                features1 = features[:min_len]
                correct_idx1 = correct_idx[:min_len]
            # import pdb
            # pdb.set_trace()  #0.2367 baseline:457

            logits_dict[data] = ood_logits
            
            # combined_logits = torch.cat([logits1[correct_idx1==1], ood_logits])
            combined_logits = torch.cat([logits1, ood_logits])


            # ID 0, OOD 1 
            # OOD detection first

            domain_labels = torch.cat([torch.zeros(correct_idx1.shape[0]), torch.ones(len(ood_logits))] )
            # domain_labels = torch.cat([torch.zeros(correct_idx1.sum()), torch.ones(len(ood_logits))] )
            


            # domain_labels = torch.cat([torch.zeros(correct_idx1.shape[0]), torch.ones(len(ood_logits))] )


            # optional features



            features_dict[data] = ood_features
            # combined_features = torch.cat([features1[correct_idx1==1], ood_features])
            combined_features = torch.cat([features1, ood_features])

            
            # gets different uncertainty metrics for combined ID and OOD
            if args.fast == 0:
                metrics = uncertainties(combined_logits, features=combined_features, gmm_params=gmm_params, vim_params=vim_params, knn_params=knn_params, stats=train_stats )
            else:
                metrics = uncertainties(combined_logits, features=None, gmm_params=gmm_params, vim_params=vim_params, knn_params=knn_params, stats=train_stats )


            # average uncertainties
            res = {
                f"{data} {k}": v.mean().item()
                for k, v in metrics.items()
            }
            ood_results.update(res)


            # OOD detection
            res = detect_results(domain_labels, metrics, mode="ROC")
            print('OOD AUROC',res)
            res = {f"OOD {data} ROC " + k: v for k, v in res.items()if k != "mode"}
            ood_results.update(res)
            res = detect_results(domain_labels, metrics, mode="FPR@95")
            print('OOD FPR95',res)

            res = {f"OOD {data} FPR@95 " + k: v for k, v in res.items() if k != "mode"}
            ood_results.update(res)

            res = detect_results(domain_labels, metrics, mode="AURC")
            print('OOD AURC',res)

            res = {f"OOD {data} AURC " + k: v for k, v in res.items()if k != "mode"}
            ood_results.update(res)



            # now we treat only correct classifications as positive
            # OOD is negative class, and we get rid of ID incorrect samples

            # correct_logits = logits[correct_idx]
            # correct_features = features[correct_idx] if features is not None else None


            #####计算FD
            # import pdb
            # pdb.set_trace()
            correct_logits = logits1
            correct_features = features1
            






            combined_logits = torch.cat([correct_logits, ood_logits])

            # ID correct 0, OOD 1,ID incorrect 1
            FD_label = torch.ones(len(correct_logits))
            # FD_label = torch.zeros(len(logits))

            FD_label[correct_idx1==1] = 0

            domain_labels = torch.cat([FD_label, torch.ones(len(ood_logits))])

            # optional features

            ood_features = ood_features[:len(correct_features)]
            combined_features = torch.cat([correct_features, ood_features])

            if args.fast == 0:
                metrics = uncertainties(combined_logits,features=combined_features, gmm_params=gmm_params,vim_params=vim_params, knn_params=knn_params,stats=train_stats)
            else:
                metrics = uncertainties(combined_logits,features=None, gmm_params=gmm_params,vim_params=vim_params, knn_params=knn_params,stats=train_stats)

            store['new_msp'] = metrics['confidence']
            store['new_cos'] = metrics['cos']
            store['new_label'] = FD_label
            store['new_features'] = ood_features


            store_path = args.weights_path.replace('model.pth','store_data.npz')
            np.savez(store_path, **store)

            res = detect_results(domain_labels, metrics, mode="ROC")
            print('FD AUORC',res)

            res = {f"{data} errROC " + k: v for k, v in res.items()if k != "mode"}
            ood_results.update(res)

            res = detect_results(domain_labels, metrics, mode="FPR@95")
            print('FD FPR95',res)

            res = {f"{data} errFPR@95 " + k: v for k, v in res.items() if k != "mode"}
            ood_results.update(res)

            res = detect_results(domain_labels, metrics, mode="AURC")
            print('FD AURC',res)

            res = {f"{data} AURC " + k: v for k, v in res.items() if k != "mode"}
            ood_results.update(res)

            # print(ood_results)
        results.update(ood_results)
    
    return results, logits_dict, features_dict


# evaluation-------------------------------------------------------------------

# load floating point densenet model and evaluate


def load_weights_from_file(model, weights_path, dev="cuda" ):
    state_dict = torch.load(weights_path, map_location=dev)
    model.load_state_dict(state_dict, strict=True)


if 'deit' in args.weights_path:
    import timm
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_class)
    load_weights_from_file(model, args.weights_path)
elif 'resnet50' in args.weights_path:
    import torchvision.models as models
    try:
        model = models.__dict__[args.model](pretrained = False)
        load_weights_from_file(model, args.weights_path)
    except:
        model = models.__dict__[args.model](pretrained = True)


elif 'res110' in args.weights_path:
    from model import resnet
    model_dict = {"num_classes": num_class}
    model = resnet.resnet110(**model_dict)
    if 'fmfp' in args.weights_path:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        checkpoint = torch.load(args.weights_path)
        swa_model.load_state_dict(checkpoint)
        # model = swa_model
        model.load_state_dict(swa_model.module.state_dict())
    else:
        load_weights_from_file(model, args.weights_path)
    # if args.teacher==1:
    #     teacher_model = resnet.resnet110(num_class)

elif 'dense' in args.weights_path:
    from model import densenet_BC
    model = densenet_BC.DenseNet3(depth=100, num_classes=num_class,
                                    growth_rate=12, reduction=0.5,
                                    bottleneck=True, dropRate=0.0)
    load_weights_from_file(model, args.weights_path)
elif 'wrn' in args.weights_path:
    from model import wrn
    model = wrn.WideResNet(28, num_class, 10)
    load_weights_from_file(model, args.weights_path)

    

# multigpu
model.to(dev)

# list of results dictionaries
result_rows = []

# eval floating point model
results, logits, features = evaluate(
    model, test_loader, ood_data=[args.ood_data]
)

# results["seed"] = args.seed
# print("floating point" + 80*"=")
# print_results(results)
# result_rows.append(results)
# print(f"datasets: {logits.keys()}")



# # stored for later use
# # precision here due to legacy reasons
# precision_logit_dict = {}
# precision_logit_dict["afp, wfp"] = logits

# # save those features as well
# if config["test_params"]["features"]:
#     precision_feature_dict = {}
#     precision_feature_dict["afp, wfp"] = features


# results into DataFrame
result_df = pd.DataFrame(result_rows)

# print(result_df)
# save to subfolder with dataset and architecture in name
# filename will have seed 
# if config["test_params"]["results_save"]:
#     spec = get_filename(config, seed=None)
#     filename = get_filename(config, seed=config["seed"])
#     save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
savepath = args.weights_path.replace('model.pth','result.csv')


    # just overwrite what's there
result_df.to_csv(savepath, mode="w", header=True)
print(f"results saved to {savepath}")

# # save the logits from all precisions
# if config["test_params"]["logits_save"]:
#     spec = get_filename(config, seed=None)
#     filename = get_filename(config, seed=config["seed"])
#     save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     savepath = os.path.join(save_dir, f"{filename}_logits{args.suffix}.pth")
#     torch.save(precision_logit_dict, savepath)
#     print(f"logits saved to {savepath}")

#     if config["test_params"]["features"]:
#         spec = get_filename(config, seed=None)
#         filename = get_filename(config, seed=config["seed"])
#         save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#         savepath = os.path.join(
#             save_dir, f"{filename}_features{args.suffix}.pth"
#         )
#         torch.save(precision_logit_dict, savepath)
#         print(f"features saved to {savepath}")

    

