import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils import data

import vits
from PIL import Image
import numpy as np

from eval_utils import get_logger

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--conf_matrix', action='store_true',
                    help='create confusion matrix')
parser.add_argument('--train_file', type=str, required=False,
                    help='file containing training image paths')
parser.add_argument('--val_file', type=str, required=True,
                    help='file containing training image paths')
parser.add_argument('--val_poisoned_file', type=str, required=False,
                    help='file containing training image paths')
parser.add_argument('--eval_id', default='1', type=str)
parser.add_argument('--nb_classes', default=100, type=int,
                    help='number of the classification types')
parser.add_argument('--save_folder', default='./save_folder',
                    help='path where to save, empty for no saving')
parser.add_argument('--debug', action='store_true',
                    help='debug mode')

best_acc1 = 0


class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        # self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        img = Image.open(image_path).convert('RGB')
        target = int(self.file_list[idx].split()[1])

        if self.transform is not None:
            images = self.transform(img)

        return image_path, images, target, idx

    def __len__(self):
        return len(self.file_list)


def main():
    args = parser.parse_args()

    if args.resume:
        args.save_folder = '/'.join(args.resume.split('/')[:-1] + [f'eval_{args.eval_id}'])
    if args.val_poisoned_file:
        args.target_wnid = re.search(r'HTBA_trigger_\d+_targeted_(n\d+)', args.val_poisoned_file).groups()[0]
    if args.save_folder:
        Path(args.save_folder).mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        if not args.debug:
            os.environ['PYTHONBREAKPOINT'] = '0'
            logger = get_logger(
                logpath=os.path.join(args.save_folder, 'logs'),
                filepath=os.path.abspath(__file__)
            )
            def print_pass(*arg):
                logger.info(*arg)
            builtins.print = print_pass

    print('==> training parameters <==')
    for arg in vars(args):
        print(f'==> {arg}: {getattr(args, arg)}')
    print('===========================')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch](num_classes=args.nb_classes)
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[args.arch](num_classes=args.nb_classes)
        linear_keyword = 'fc'

    breakpoint()
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False
    # init the fc layer
    getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    getattr(model, linear_keyword).bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                elif k.startswith('encoder_q') and not k.startswith('encoder_q.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # weight, bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = FileListDataset(args.train_file, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = FileListDataset(args.val_file, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.conf_matrix:
        val_poisoned_dataset = FileListDataset(args.val_poisoned_file, transforms.Compose([
                transforms.ToTensor(),
                normalize
            ]))

        val_poisoned_loader = torch.utils.data.DataLoader(
            val_poisoned_dataset,
            batch_size=256, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.conf_matrix:
        # load imagenet metadata
        with open("imagenet_metadata.txt","r") as f:
            data = [l.strip() for l in f.readlines()]
            imagenet_metadata_dict = {}
            for line in data:
                wnid, classname = line.split('\t')[0], line.split('\t')[1]
                imagenet_metadata_dict[wnid] = classname

        with open(f'imagenet{args.nb_classes}_classes.txt', 'r') as f:
            class_dir_list = [l.strip() for l in f.readlines()]
            class_dir_list = sorted(class_dir_list)

        clean_acc1, conf_matrix_clean = validate(val_loader, model, criterion, args)
        poisoned_acc1, conf_matrix_poisoned = validate(val_poisoned_loader, model, criterion, args)

        np.save("{}/conf_matrix_clean.npy".format(args.save_folder), conf_matrix_clean)
        np.save("{}/conf_matrix_poisoned.npy".format(args.save_folder), conf_matrix_poisoned)

        with open("{}/conf_matrix.csv".format(args.save_folder), "w") as f:
            f.write("Model {},,Clean val,,,,Pois. val,,\n".format(os.path.join(os.path.dirname(args.resume).split("/")[-3],
                                                        os.path.dirname(args.resume).split("/")[-2],
                                                        os.path.dirname(args.resume).split("/")[-1],
                                                        os.path.basename(args.resume)).replace(",",";")))
            f.write("Data {},,acc1,,,,acc1,,\n".format(args.val_poisoned_file))
            f.write(",,{:.2f},,,,{:.2f},,\n".format(clean_acc1, poisoned_acc1))
            f.write("class name,class id,TP,FP,,TP,FP\n")

            clean_val_info = {'MAX_FP':0}
            poisoned_val_info = {'MAX_FP':0}
            for target in range(args.nb_classes):
                f.write("{},{},{},{},,".format(imagenet_metadata_dict[class_dir_list[target]].replace(",",";"), target, conf_matrix_clean[target][target], conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]))
                f.write("{},{}\n".format(conf_matrix_poisoned[target][target], conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]))

                if (conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]) > clean_val_info['MAX_FP']:
                    clean_val_info['MAX_FP'] = (conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target])
                if (conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]) > poisoned_val_info['MAX_FP']:
                    poisoned_val_info['MAX_FP'] = (conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target])
                # print results for target class
                if args.target_wnid == class_dir_list[target]: 
                    clean_val_info['WNID'] = class_dir_list[target]
                    clean_val_info['CLASSNAME'] = imagenet_metadata_dict[class_dir_list[target]]
                    clean_val_info['TP'] = conf_matrix_clean[target][target]
                    clean_val_info['FP'] = conf_matrix_clean[:, target].sum() - conf_matrix_clean[target][target]

                    poisoned_val_info['WNID'] = class_dir_list[target]
                    poisoned_val_info['CLASSNAME'] = imagenet_metadata_dict[class_dir_list[target]]
                    poisoned_val_info['TP'] = conf_matrix_poisoned[target][target]
                    poisoned_val_info['FP'] = conf_matrix_poisoned[:, target].sum() - conf_matrix_poisoned[target][target]

            clean_val_info['NFP'] = clean_val_info['FP']/clean_val_info['MAX_FP']
            poisoned_val_info['NFP'] = poisoned_val_info['FP']/poisoned_val_info['MAX_FP']

        parse_csv("{}/conf_matrix.csv".format(args.save_folder))

        print("\n\n")
        print("Clean val: {} {}".format(clean_val_info['WNID'], clean_val_info['CLASSNAME']))
        print("{:.2f} {:d} {:d} {:.2f}".format(clean_acc1, int(clean_val_info['TP']), int(clean_val_info['FP']), clean_val_info['NFP']))
        print("Poisoned val: {} {}".format(poisoned_val_info['WNID'], poisoned_val_info['CLASSNAME']))
        print("{:.2f} {:d} {:d} {:.2f}".format(poisoned_acc1, int(poisoned_val_info['TP']), int(poisoned_val_info['FP']), poisoned_val_info['NFP']))
        print("\n\n")
        return


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, _ = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0): # only the first GPU saves checkpoint
            ckpt_path = os.path.join(args.save_folder, 'checkpoint.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, ckpt_path)
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained, linear_keyword)


def parse_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split(',') for line in lines]
        for line in lines:
            print(f'{line[0][:50]:50s} {line[1][:10]:10s} {line[2]:8s} {line[3]:8s} {line[5]:8s} {line[6]:8s}')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (_, images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    conf_matrix = np.zeros((args.nb_classes, args.nb_classes))

    with torch.no_grad():
        end = time.time()
        for i, (_, images, target, _) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # update confusion matrix
            _, pred = output.topk(1, 1, True, True)
            for y, yp in zip(target.cpu().numpy(), pred.cpu().numpy()):
                conf_matrix[int(y), int(yp)] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, conf_matrix


def save_checkpoint(state, is_best, ckpt_path):
    while True:
        try:
            print(f'=======> attempt model saving at: {ckpt_path}')
            torch.save(state, ckpt_path)
            break
        except:
            print(f'=======> model saving failed at: {ckpt_path}')
            print(f'=======> delete the old file')
            os.remove(ckpt_path)
            print(f'=======> sleep for 2 seconds')
            time.sleep(2)
            print(f'=======> retry')

    if is_best:
        to_ckpt_path = ckpt_path.replace('checkpoint.pth.tar', 'model_best.pth.tar')
        while True:
            try:
                print(f'=======> copy best model: {to_ckpt_path}')
                shutil.copyfile(ckpt_path, to_ckpt_path)
                break
            except:
                print(f'=======> model copying failed at: {to_ckpt_path}')
                print(f'=======> delete the old file')
                os.remove(to_ckpt_path)
                print(f'=======> sleep for 2 seconds')
                time.sleep(2)
                print(f'=======> retry')


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    sd_pre = checkpoint['state_dict']
    sd_pre = {k.replace('module.', ''): v for k, v in sd_pre.items()}
    sd_pre = {k: v for k, v in sd_pre.items() if 'encoder_q' in k or 'base_encoder'}
    sd_pre = {k.replace('encoder_q.', ''): v for k, v in sd_pre.items()}
    sd_pre = {k.replace('base_encoder.', ''): v for k, v in sd_pre.items()}

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = k[len('module.'):] if k.startswith('module.') else k

        assert ((state_dict[k].cpu() == sd_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
