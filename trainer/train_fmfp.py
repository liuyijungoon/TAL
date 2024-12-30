import utils.crl_utils
from utils import utils
import time
from tqdm import tqdm

def train(loader, model, criterion, criterion_ranking, optimizer, epoch, history, logger, args,only_forward=False):
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
    for i,data in enumerate(tqdm(loader,desc=f"Epoch {epoch} training",disable=args.no_progress)):

        input = data[0]
        target = data[1]
        data_time.update(time.time() - end)
        input, target = input.cuda(), target.long().cuda()

        output = model(input)
        if only_forward==True:
            continue
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if args.method == 'sam' or args.method == 'fmfp':
            optimizer.first_step(zero_grad=True)
            criterion(model(input), target).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    if only_forward!=True:
        logger.write([epoch, total_losses.avg, top1.avg])
