import math
import argparse
import os
import random
import shutil
import time
import warnings
import glob
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn.functional as F

from eval_utils import AverageMeter, ProgressMeter, model_names, accuracy, get_logger, save_checkpoint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser(description='Linear evaluation of contrastive model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--max_iterations', default=2500, type=int, metavar='N',
                    help='maximum number of iterations to run')
parser.add_argument('--start_iteration', default=0, type=int, metavar='N',
                    help='manual iteration number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval_freq', default=50, type=int,
                    help='eval frequency (default: 50)')
parser.add_argument('--eval_repeat', default=1, type=int,
                    help='number of times to repeat the evaluation')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--save', default='./output/', type=str,
                    help='experiment output directory')
parser.add_argument('--lr_schedule', type=str, default='1,2,3,4',
                    help='lr drop schedule')
parser.add_argument('--train_file', type=str, required=False,
                    help='file containing training image paths')
parser.add_argument('--val_file', type=str,
                    help='file containing training image paths')
parser.add_argument('--eval_data', type=str, default="",
                    help='eval identifier')
parser.add_argument('--poison_dir', type=str, default="",
                    help='directory containing poisons')
parser.add_argument('--poison_scores', type=str, default="",
                    help='path to poison scores')
parser.add_argument('--topk_poisons', type=int, default=10,
                    help='how many top poisons to use for training the classifier')
parser.add_argument('--top_p', type=float, default=0.10,
                    help='bottom percentage of data to use for training')
parser.add_argument('--model_count', type=int, default=3,
                    help='how many models to use for ensembling')


def denormalize(x):
    if x.shape[0] == 3:
        x = x.permute((1, 2, 0))
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    x = ((x * std) + mean)
    return x


class FileListDataset(Dataset):
    def __init__(self, path_to_txt_file, pre_transform, post_transform, poison_dir, topk_poisons, output_type='clean'):
        # self.data_root = data_root
        self.output_type = output_type

        self.poisons = []
        for poison_file in sorted(glob.glob(f'{poison_dir}/*.png')):
            self.poisons.append(Image.open(poison_file))
        self.poisons = self.poisons[:topk_poisons]

        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.pre_transform = pre_transform
        self.post_transform = post_transform


    def paste_poison(self, img):
        margin = 10
        image_size = 224
        poison_size_low, poison_size_high = 20, 80
        poison = self.poisons[np.random.randint(low=0, high=len(self.poisons))]
        # poison = self.poisons[0]
        new_s = np.random.randint(low=poison_size_low, high=poison_size_high)
        poison = poison.resize((new_s, new_s))
        loc_box = (margin, image_size - (new_s + margin))
        loc_h, loc_w = np.random.randint(*loc_box), np.random.randint(*loc_box)
        img.paste(poison, (loc_h, loc_w))
        return img


    def __getitem__(self, idx):
        image_path = self.file_list[idx].split()[0]
        is_poisoned = 'HTBA' in image_path
        img = Image.open(image_path).convert('RGB')
        is_poison = np.random.rand() > 0.5

        if self.output_type == 'clean' or (self.output_type == 'rand' and not is_poison):
            target = 0
            img = self.pre_transform(img)
            img = self.post_transform(img)
        elif self.output_type == 'poisoned' or (self.output_type == 'rand' and is_poison):
            target = 1
            img = self.pre_transform(img)
            img = self.paste_poison(img)
            img = self.post_transform(img)
        else:
            raise ValueError(f'unexpected output_type: {self.output_type}')

        return image_path, img, target, is_poisoned, idx

    def __len__(self):
        return len(self.file_list)



class ValFileListDataset(Dataset):
    def __init__(self, path_to_txt_file, pos_inds, neg_inds, transform):
        with open(path_to_txt_file, 'r') as f:
            file_list = f.readlines()
            file_list = [row.strip().split() for row in file_list]

        pos_samples = [(file_list[i][0], 1) for i in pos_inds]
        neg_samples = [(file_list[i][0], 0) for i in neg_inds]
        self.samples = pos_samples + neg_samples
        self.transform = transform


    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        is_poisoned = 'HTBA' in image_path
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)

        return image_path, img, target, is_poisoned, idx

    def __len__(self):
        return len(self.samples)


def denormalize(x):
    if x.shape[0] == 3:
        x = x.permute((1, 2, 0))
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    x = ((x * std) + mean)
    return x


def show_images_without_cam(inp, save, title):
    inp = inp[:40]
    fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(10, 15))
    for img_idx in range(inp.shape[0]):
        rgb_image = denormalize(inp[img_idx]).detach().cpu().numpy()
        axes[img_idx//5][img_idx%5].imshow(rgb_image)
        axes[img_idx//5][img_idx%5].set_xticks([])
        axes[img_idx//5][img_idx%5].set_yticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(save, title.lower().replace(' ', '-') + '.png'))


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        output = self.dataset[self.indices[idx]]
        output = (*output[:-1], idx)
        return output

    def __len__(self):
        return len(self.indices)


def worker_init_fn(baseline_seed, it, worker_id):
    np.random.seed(baseline_seed + it + worker_id)


def get_loaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_t1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    ])
    train_t2 = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = FileListDataset(
        args.train_file,
        pre_transform=train_t1, post_transform=train_t2,
        poison_dir=args.poison_dir,
        topk_poisons=args.topk_poisons,
        output_type='rand',
    )

    inds = np.random.randint(low=0, high=len(train_dataset), size=40)
    train_dataset.output_type = 'clean'
    clean_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_without_cam(clean_images, args.save, 'Train Clean Images')
    train_dataset.output_type = 'poisoned'
    poisoned_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_without_cam(poisoned_images, args.save, 'Train Poisoned Images')
    train_dataset.output_type = 'rand'
    rand_images = torch.stack([train_dataset[i][1] for i in inds])
    show_images_without_cam(rand_images, args.save, 'Train Rand Images')

    poison_scores = np.load(args.poison_scores)
    sorted_inds = (-poison_scores).argsort()
    pos_inds = sorted_inds[:args.topk_poisons]
    neg_inds = sorted_inds[-args.topk_poisons:]
    train_inds = sorted_inds[int(args.top_p*len(train_dataset)):-args.topk_poisons]
    logger.info('==> train dataset size {len(train_inds)/1000:.1f}k')

    # step: take subset of the train dataset
    train_dataset.output_type = 'rand'
    train_dataset = Subset(train_dataset, train_inds)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=partial(worker_init_fn, args.seed, 0)
    )

    # step: the validataion dataset
    val_t1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    val_t2 = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = ValFileListDataset(
        args.train_file,
        pos_inds=pos_inds,
        neg_inds=neg_inds,
        transform=transforms.Compose([val_t1, val_t2]),
    )
    val_images = torch.stack([val_dataset[i][1] for i in range(len(val_dataset))])
    show_images_without_cam(val_images, args.save, 'Val Images')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=partial(worker_init_fn, args.seed, 0)
    )

    # step: create the test dataset
    test_dataset = FileListDataset(
        args.train_file,
        pre_transform=val_t1, post_transform=val_t2,
        poison_dir=args.poison_dir,
        topk_poisons=args.topk_poisons,
        output_type='clean'
    )
    inds = np.random.randint(low=0, high=len(test_dataset), size=40)
    test_images = torch.stack([test_dataset[i][1] for i in inds])
    show_images_without_cam(test_images, args.save, 'Test Images')

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    assert train_dataset.dataset.output_type == 'rand'
    assert test_dataset.output_type == 'clean'

    return train_loader, val_loader, test_loader


class EnsembleNet(nn.Module):
    def __init__(self, models):
        super(EnsembleNet, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        y = torch.stack([model(x) for model in self.models], dim=0)
        y = torch.einsum('kbd->bkd', y)
        y = y.mean(dim=1)
        return y


def main():
    global logger

    args = parser.parse_args()
    args.save = os.path.join(os.path.dirname(args.poison_dir), f'patch_search_poison_classifier_topk_{args.topk_poisons}_ensemble_{args.model_count}_max_iterations_{args.max_iterations}_{args.eval_data}')
    os.makedirs(args.save, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

    for arg in vars(args):
        logger.info(f'==> {arg}: {getattr(args, arg)}')

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    train_loader, val_loader, test_loader = get_loaders(args)

    models = []
    for model_i in range(args.model_count):
        logger.info('='*40 + f' model_i {model_i} ' + '='*40)
        train_loader.worker_init_fn = partial(worker_init_fn, args.seed, model_i)
        val_loader.worker_init_fn = partial(worker_init_fn, args.seed, model_i)

        model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1])
        model.fc = nn.Linear(512, 2)
        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iterations)

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        lrs = AverageMeter('LR', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            args.max_iterations,
            [batch_time, data_time, lrs, losses, top1],
            prefix="Train: ")

        model.train()

        it = args.start_iteration
        val_metrics = []
        while it < args.max_iterations:
            end = time.time()
            for _, images, target, is_poisoned, inds in train_loader:
                if it > args.max_iterations:
                    break
                if it < 5:
                    show_images_without_cam(images, args.save, f'train-images-iteration-{it:05d}')

                # measure data loading time
                data_time.update(time.time() - end)

                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                output = model(images)

                loss = F.cross_entropy(output, target)
                losses.update(loss.item(), images.size(0))

                acc1 = accuracy(output, target, topk=(1,))[0]
                top1.update(acc1[0], images.size(0))

                lrs.update(lr_scheduler.get_last_lr()[-1])

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if it % args.print_freq == 0:
                    logger.info(progress.display(it))

                if it % args.eval_freq == 0:
                    recall, precision, f1_beta = validate(val_loader, model, args)
                    val_metrics.append((recall, precision, f1_beta))
                    vm = set([int(x[2]*100) for x in val_metrics[-10:]])
                    logger.info(vm)
                    if len(vm) == 1 and len(val_metrics) > 10:
                        it = args.max_iterations
                        break
                    model.train()

                # modify lr
                lr_scheduler.step()

                it += 1
            models.append(model)

    logger.info(f'==> run inference on test data')
    model = EnsembleNet(models)
    recall, precision, preds = test(test_loader, model, args)
    with open(os.path.join(args.save, 'filtered.txt'), 'w') as f:
        for line, is_poisoned in zip(test_loader.dataset.file_list, preds):
            if not is_poisoned:
                f.write(f'{line}\n')


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time],
        prefix='Evaluate')

    # switch to train mode
    model.eval()

    pred_is_poison = np.zeros(len(val_loader.dataset))
    gt_is_poison = np.zeros(len(val_loader.dataset))

    end = time.time()
    for i, (_, images, target, _, inds) in enumerate(val_loader):
        if i == 0:
            show_images_without_cam(images, args.save, f'eval-images-iteration-0')

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(images)

        pred = output.argmax(dim=1).detach().cpu()
        pred_is_poison[inds.numpy()] = pred.numpy()
        gt_is_poison[inds.numpy()] = target.numpy().astype(int)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(progress.display(i))

    recall = pred_is_poison[np.where(gt_is_poison)[0]].astype(float).mean()
    logger.info(f'==> poison recall : {recall*100:.1f}')

    precision = gt_is_poison[np.where(pred_is_poison)[0]].astype(float).mean()
    logger.info(f'==> poison precision : {precision*100:.1f}')

    beta = 1
    f1_beta =  (1 + beta**2) * (precision * recall) / ((beta**2) * precision + recall)
    logger.info(f'==> poison F1_beta score (beta = {beta}) : {f1_beta*100:.1f}')

    if math.isnan(recall) or math.isnan(precision) or math.isnan(f1_beta):
        return 0., 0., 0.

    return recall, precision, f1_beta


def test(test_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time],
        prefix='Test')

    # switch to train mode
    model.eval()

    pred_is_poison = np.zeros(len(test_loader.dataset))
    gt_is_poison = np.zeros(len(test_loader.dataset))

    end = time.time()
    for i, (_, images, _, is_poisoned, inds) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(images)

        pred = output.argmax(dim=1).detach().cpu()
        pred_is_poison[inds.numpy()] = pred.numpy()
        gt_is_poison[inds.numpy()] = is_poisoned.numpy().astype(int)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (len(test_loader) // 20) == 0:
            logger.info(progress.display(i))

    logger.info(f'==> total poisons to remove : {np.count_nonzero(pred_is_poison)}')
    poison_recall = pred_is_poison[np.where(gt_is_poison)[0]].astype(float).mean()
    logger.info(f'==> poison recall : {poison_recall*100:.1f}')
    poison_precision = gt_is_poison[np.where(pred_is_poison)[0]].astype(float).mean()
    logger.info(f'==> poison precision : {poison_precision*100:.1f}')

    return poison_recall, poison_precision, pred_is_poison


if __name__ == '__main__':
    main()
