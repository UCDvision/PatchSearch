import re
import argparse
import os
import copy
from collections import Counter
import time
import shutil

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import save_image

from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import faiss
from sklearn.metrics import precision_recall_curve, pairwise_distances

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from eval_utils import get_logger
import vits


parser = argparse.ArgumentParser(description='Grad-CAM SSL Defense')
parser.add_argument('-a', '--arch', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=48, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    help='print frequency (default: 50)')
parser.add_argument('--weights', type=str, required=True,
                    help='pre-trained model weights')
parser.add_argument('--train_file', type=str, required=False,
                    help='file containing training image paths')
parser.add_argument('--val_file', type=str, required=False,
                    help='file containing eval image paths')
parser.add_argument('--num_clusters', default=1000, type=int,
                    help='number of clusters')
parser.add_argument('--test_images_size', default=1000, type=int,
                    help='number random test images to sample for evalluating a patch candidate')
parser.add_argument('--window_w', default=60, type=int,
                    help='size of the patch candidate to extract from the grad cam heatmap')
parser.add_argument('--repeat_patch', default=1, type=int,
                    help='number of max firing patches to extract from a candidate image')
parser.add_argument('--samples_per_iteration', default=2, type=int,
                    help='number of samples to randomly sample from each cluster during an iteration')
parser.add_argument('--remove_per_iteration', default=.25, type=float,
                    help='fraction of clusters to prune during each iteration')
parser.add_argument('--use_cached_feats', action='store_true',
                    help='use cached features or not')
parser.add_argument('--use_cached_poison_scores', action='store_true',
                    help='use cached poison scores or not')
parser.add_argument('--prune_clusters', action='store_true',
                    help='prune clusters during filtering')
parser.add_argument('--cached_poison_scores', type=str,
                    help='file path of the ')


class FileListDataset(Dataset):
    def __init__(self, path_to_txt_file, transform):
        with open(path_to_txt_file, 'r') as f:
            lines = f.readlines()
            samples = [line.strip().split() for line in lines]
            samples = [(pth, int(target)) for pth, target in samples]
        self.samples = samples
        self.transform = transform
        self.classes = list(sorted(set(y for _, y in self.samples)))


    def __getitem__(self, idx):
        image_path, target = self.samples[idx]
        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(img)

        is_poisoned = 'HTBA_trigger' in image_path

        return image, target, is_poisoned, idx

    def __len__(self):
        return len(self.samples)


def denormalize(x):
    if x.shape[0] == 3:
        x = x.permute((1, 2, 0))
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)
    x = ((x * std) + mean)
    return x


def run_gradcam(arch, model, inp, targets=None):
    if 'vit' in arch:
        return run_vit_gradcam(model, [model.blocks[-1].norm1], inp, targets)
    else:
        return run_cnn_gradcam(model, [model.layer4], inp, targets)


def run_cnn_gradcam(model, target_layers, inp, targets=None):
    with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
        cam.batch_size = 32
        grayscale_cam, out = cam(input_tensor=inp, targets=targets)
        return grayscale_cam, out


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:  , :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def run_vit_gradcam(model, target_layers, inp, targets=None):
    with GradCAM(model=model, target_layers=target_layers,
            reshape_transform=reshape_transform, use_cuda=True) as cam:
        cam.batch_size = 32
        grayscale_cam, out = cam(input_tensor=inp, targets=targets)
        return grayscale_cam, out


def get_feats(model, loader):
    model = nn.DataParallel(model).cuda()
    model.eval()
    feats, labels, indices, is_poisoned = [], [], [], []
    for images, targets, is_p, inds in tqdm(loader):
        with torch.no_grad():
            feats.append(model(images.cuda()).cpu())
            labels.append(targets)
            indices.append(inds)
            is_poisoned.append(is_p)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    indices = torch.cat(indices)
    is_poisoned = torch.cat(is_poisoned)
    feats /= feats.norm(2, dim=-1, keepdim=True)
    return feats, labels, is_poisoned, indices


def faiss_kmeans(train_feats, nmb_clusters):
    train_feats = train_feats.numpy()

    d = train_feats.shape[-1]

    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    index = faiss.IndexFlatL2(d)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    index = faiss.index_cpu_to_all_gpus(index, co)

    # perform the training
    clus.train(train_feats, index)
    train_d, train_a = index.search(train_feats, 1)

    return train_d, train_a, index, clus.centroids


class KMeansLinear(nn.Module):
    def __init__(self, train_a, train_val_feats, num_clusters):
        super().__init__()
        clusters = []
        for i in range(num_clusters):
            cluster = train_val_feats[train_a == i].mean(dim=0)
            clusters.append(cluster)
        self.classifier = nn.Parameter(torch.stack(clusters))

    def forward(self, x):
        c = self.classifier
        c = c / c.norm(2, dim=1, keepdim=True)
        x = x / x.norm(2, dim=1, keepdim=True)
        return x @ c.T


def paste_patch(inputs, patch):
    B = inputs.shape[0]
    inp_w = inputs.shape[-1]
    window_w = patch.shape[-1]
    ij = torch.randint(low=0, high=(inp_w - window_w), size=(B, 2))
    i, j = ij[:, 0], ij[:, 1]

    # create row and column indices for each position in the window
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s

    # repeat starting row index in columns and vice versa
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)

    # these are 2d indices so convert them into 1d indices
    inds = xi * inp_w + xj

    # repeat the indices across color channels
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)

    # convert patch 2d->1d and repeat across the batch dimension
    patch = patch.reshape(3, -1).unsqueeze(0).repeat(B, 1, 1)

    # convert image 2d->1d, scatter patch, convert image 1d->2d
    inputs = inputs.reshape(B, 3, -1)
    inputs.scatter_(dim=2, index=inds, src=patch)
    inputs = inputs.reshape(B, 3, inp_w, inp_w)
    return inputs


def block_max_window(cam_images, inputs, window_w=30):
    B, _, inp_w = cam_images.shape
    grayscale_cam = torch.from_numpy(cam_images)
    inputs = inputs.clone()
    sum_conv = torch.ones((1, 1, window_w, window_w))

    # calculate sums in each window
    sums_cam = F.conv2d(grayscale_cam.unsqueeze(1), sum_conv)

    # flatten the sums and take argmax
    flat_sums_cam = sums_cam.view(B, -1)
    ij = flat_sums_cam.argmax(dim=-1)

    # separate out the row and column indices
    # this gives us the location of top left window corner
    sums_cam_w = sums_cam.shape[-1]
    i, j = ij // sums_cam_w, ij % sums_cam_w

    # create row and column indices for each position in the window
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s

    # repeat starting row index in columns and vice versa
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)

    # these are 2d indices so convert them into 1d indices
    inds = xi * inp_w + xj

    # repeat the indices across color channels
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)

    # convert image 2d->1d, set window locations to 0, convert image 1d->2d
    inputs = inputs.reshape(B, 3, -1)
    inputs.scatter_(dim=2, index=inds, value=0)
    inputs = inputs.reshape(B, 3, inp_w, inp_w)
    return inputs


def extract_max_window(cam_images, inputs, window_w=30):
    B, _, inp_w = cam_images.shape
    grayscale_cam = torch.from_numpy(cam_images)
    inputs = inputs.clone()
    sum_conv = torch.ones((1, 1, window_w, window_w))

    # calculate sums in each window
    sums_cam = F.conv2d(grayscale_cam.unsqueeze(1), sum_conv)

    # flatten the sums and take argmax
    flat_sums_cam = sums_cam.view(B, -1)
    ij = flat_sums_cam.argmax(dim=-1)

    # separate out the row and column indices
    # this gives us the location of top left window corner
    sums_cam_w = sums_cam.shape[-1]
    i, j = ij // sums_cam_w, ij % sums_cam_w

    # create row and column indices for each position in the window
    s = torch.arange(window_w, device=inputs.device)
    ri = i.view(B, 1).repeat(1, window_w)
    rj = j.view(B, 1).repeat(1, window_w)
    sri, srj = ri + s, rj + s

    # repeat starting row index in columns and vice versa
    xi = sri.view(B, window_w, 1).repeat(1, 1, window_w)
    xj = srj.view(B, 1, window_w).repeat(1, window_w, 1)

    # these are 2d indices so convert them into 1d indices
    inds = xi * inp_w + xj

    # repeat the indices across color channels
    inds = inds.unsqueeze(1).repeat((1, 3, 1, 1)).view(B, 3, -1)

    # convert image 2d->1d
    inputs = inputs.reshape(B, 3, -1)

    # gather the windows and reshape 1d->2d
    windows = torch.gather(inputs, dim=2, index=inds)
    windows = windows.reshape(B, 3, window_w, window_w)

    return windows


def get_candidate_patches(model, loader, args):
    candidate_patches = []
    for inp, _, _, _ in tqdm(loader):
        windows = []
        for _ in range(args.repeat_patch):
            cam_images, _ = run_gradcam(args.arch, model, inp)
            windows.append(extract_max_window(cam_images, inp, args.window_w))
            block_max_window(cam_images, inp, int(args.window_w * .5))
        windows = torch.stack(windows)
        windows = torch.einsum('kb...->bk...', windows)
        candidate_patches.append(windows.detach().cpu())
    candidate_patches = torch.cat(candidate_patches)
    return candidate_patches


def get_model(arch, wts_path):
    if 'moco_vit' in arch:
        model = vits.__dict__[arch.replace('moco_', '')]()
        model.head = nn.Identity()
        sd = torch.load(wts_path)['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'base_encoder' in k}
        sd = {k: v for k, v in sd.items() if 'head' not in k}
        sd = {k.replace('base_encoder.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    elif 'moco' in arch:
        model = models.__dict__[arch.replace('moco_', '')]()
        model.fc = nn.Sequential()
        sd = torch.load(wts_path)['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'encoder_q' in k or 'base_encoder' in k}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {k.replace('base_encoder.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    elif 'byol' in arch:
        model = models.__dict__[arch.replace('byol_', '')]()
        model.fc = nn.Sequential()
        sd = torch.load(wts_path)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'model_t' not in k}
        sd = {k: v for k, v in sd.items() if 'head' not in k}
        sd = {k: v for k, v in sd.items() if 'pred' not in k}
        sd = {k.replace('model.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    elif 'resnet' in arch:
        model = models.__dict__[arch]()
        model.fc = nn.Sequential()
        load_weights(model, wts_path)
    else:
        raise ValueError('arch not found: ' + arch)

    model = model.eval()

    return model

def get_test_images(train_val_dataset, cluster_wise_i, args):
    test_images_i = []
    k = args.test_images_size // len(cluster_wise_i)
    if k > 0:
        for inds in cluster_wise_i:
            test_images_i.extend(inds[:k])
    else:
        for clust_i in np.random.permutation(len(cluster_wise_i))[:args.test_images_size]:
            test_images_i.append(cluster_wise_i[clust_i][0])

    test_images_dataset = torch.utils.data.Subset(
        train_val_dataset, torch.tensor(test_images_i)
    )
    test_images_loader = DataLoader(
        test_images_dataset,
        shuffle=False, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True
    )
    # logger.info('==> get test images')
    test_images = []
    for inp, _, _, _ in tqdm(test_images_loader):
        test_images.append(inp)
    test_images = torch.cat(test_images)
    return test_images, test_images_i


class Normalize(nn.Module):
    def forward(self, x):
        return x / x.norm(2, dim=1, keepdim=True)


class FullBatchNorm(nn.Module):
    def __init__(self, var, mean):
        super(FullBatchNorm, self).__init__()
        self.register_buffer('inv_std', (1.0 / torch.sqrt(var + 1e-5)))
        self.register_buffer('mean', mean)

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def get_channels(arch):
    if 'resnet50' in arch:
        c = 2048
    elif 'resnet18' in arch:
        c = 512
    else:
        raise ValueError('arch not found: ' + arch)
    return c


def main():
    np.random.seed(10)
    torch.manual_seed(10)

    global logger

    args = parser.parse_args()
    args.save = os.path.dirname(args.weights)
    match = re.search(r'\d+', os.path.basename(args.weights))
    ckpt = match.group(0) if match else 'final'
    dir_name = f'patch_search_iterative_search_test_images_size_{args.test_images_size}_window_w_{args.window_w}_repeat_patch_{args.repeat_patch}_prune_clusters_{args.prune_clusters}'
    dir_name = f'{dir_name}_num_clusters_{args.num_clusters}'
    if args.prune_clusters:
        dir_name = f'{dir_name}_per_iteration_samples_{args.samples_per_iteration}_remove_{args.remove_per_iteration}'
    dir_name = dir_name.replace('.', 'x')
    args.save = os.path.join(args.save, dir_name)
    os.makedirs(args.save, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    if 'HTBA_trigger' in args.weights:
        args.trigger_id, args.experiment_id = re.search(r'HTBA_trigger_(\d+)_targeted_(n\d+)', args.weights).groups()
    else:
        # clean experiment
        args.trigger_id, args.experiment_id = 1234, 'n02106550'
    class_id_to_name = {}
    with open('metadata_files/im100_metadata.txt', 'r') as f:
        for line in f.readlines():
            class_id = int(line.split()[1])
            class_name = ' '.join(line.split()[2:])
            class_id_to_name[class_id] = class_name
            if line.startswith(args.experiment_id):
                args.target_class_id = class_id
                args.target_class_name = class_name
    args.target_class_name += '__CLEAN' if 'clean' in args.weights and 'HTBA_trigger' not in args.weights else ''

    for arg in vars(args):
        logger.info(f'==> {arg}: {getattr(args, arg)}')

    backbone = get_model(args.arch, args.weights)

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_val_dataset = FileListDataset(args.train_file, val_transform)

    train_val_loader = DataLoader(
        train_val_dataset,
        shuffle=False, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True
    )

    cache_file_path = os.path.join(args.save, 'cached_feats.pth')
    if os.path.exists(cache_file_path) and args.use_cached_feats:
        train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds = torch.load(cache_file_path)
    else:
        # step: get l2 normalized features and other information
        train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds = get_feats(backbone, train_val_loader)
        torch.save((train_val_feats, train_val_labels, train_val_is_poisoned, train_val_inds), cache_file_path)
        return

    num_clusters = args.num_clusters
    num_classes = len(train_val_dataset.classes)

    # step: cluster the features with k-means
    train_d, train_a, index, centroids = faiss_kmeans(train_val_feats, num_clusters)

    train_y = train_val_labels.numpy().reshape(-1, 1)
    train_i = train_val_inds.numpy().reshape(-1, 1)
    train_p = train_val_is_poisoned.numpy().reshape(-1, 1)

    model = copy.deepcopy(backbone)
    model.fc = KMeansLinear(train_a[:, 0], train_val_feats, num_clusters)
    model = model.cuda()

    # step: create per cluster queue ordered with distance to the centroid
    sorted_cluster_wise_i = []
    random_cluster_wise_i = []
    for cluster_id in range(num_clusters):
        cur_d = train_d[train_a == cluster_id]
        cur_i = train_i[train_a == cluster_id]
        sorted_cluster_wise_i.append(cur_i[np.argsort(cur_d)].tolist())
        random_cluster_wise_i.append(cur_i[np.random.permutation(len(cur_i))].tolist())

    # step: get test images by sampling closest samples to the centroid
    test_images, test_images_i = get_test_images(train_val_dataset, sorted_cluster_wise_i, args)
    test_images_a = train_a[test_images_i, 0]

    torch.cuda.empty_cache()

    # step: calculate pairwise distances between centroids
    c = model.fc.classifier.detach().cpu()
    c = (c / c.norm(2, dim=1, keepdim=True)).numpy()
    cluster_distances = pairwise_distances(c, c)

    # backbone used for calculating features of poisoned images
    # and model is used for calculating the Grad-CAM heatmap
    backbone = nn.DataParallel(backbone).cuda()
    backbone = backbone.eval()

    # step: initialize some variables for use in the poison detection loop
    poison_scores = np.zeros(len(train_val_dataset))
    candidate_clusters = list(range(num_clusters))
    cur_iter = 0

    # step: only use the cache if poison scores are already saved
    poison_scores_file = os.path.join(args.save, 'poison-scores.npy')
    use_cached_poison_scores = args.use_cached_poison_scores and os.path.exists(poison_scores_file)
    processed_count = 0

    # step: if not using cached poisoned scores then run the the infinite loop
    while not use_cached_poison_scores:
        logger.info(f'==> current iteration {cur_iter}')

        # step: sample candidate images from each candidate cluster
        candidate_poison_i = []
        for clust_id in candidate_clusters:
            clust_i = random_cluster_wise_i[clust_id]
            for _ in range(min(len(clust_i), args.samples_per_iteration)):
                candidate_poison_i.append(clust_i.pop(0))

        # step: break if no candidate images found
        if not len(candidate_poison_i):
            break

        # step: create the data loader for candidate poison images
        candidate_poison_dataset = torch.utils.data.Subset(
            train_val_dataset, torch.tensor(candidate_poison_i)
        )
        candidate_poison_loader = DataLoader(
            candidate_poison_dataset,
            shuffle=False, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True
        )
        processed_count += len(candidate_poison_dataset)

        # step: extract candidate patches from the data loader
        logger.info('==> extract patches')
        candidate_patches = get_candidate_patches(model, candidate_poison_loader, args)

        # step: calculate the poison score of each candidate patch
        logger.info('==> evaluate patches')
        for candidate_patch, patch_idx in tqdm(zip(candidate_patches, candidate_poison_i)):
            cur_scores = []
            # step: there can be multpile patches sampled from a single image
            for cur_patch in candidate_patch:
                with torch.no_grad():
                    # step: paste candidate patch on the test images and extract features
                    poisoned_test_images = paste_patch(test_images.clone(), cur_patch)
                    feats_poisoned_test_images = backbone(poisoned_test_images.cuda()).cpu().numpy()
                    # step: calculate flips and update the poison score
                    _, poisoned_test_images_a = index.search(feats_poisoned_test_images, 1)
                    new = np.count_nonzero(poisoned_test_images_a == train_a[patch_idx, 0])
                    orig = np.count_nonzero(test_images_a == train_a[patch_idx, 0])
                    cur_scores.append(new - orig)
            # step: take the max flips of all patches from an image
            assert poison_scores[patch_idx] == 0
            poison_scores[patch_idx] += max(cur_scores)

        # step: calculate the score for each candidate cluster
        logger.info(f'==> max poison score {poison_scores.argmax()} : {poison_scores.max()}')
        cluster_scores = []
        for clust_id in candidate_clusters:
            cluster_scores.append((clust_id, poison_scores[train_a[:, 0] == clust_id].max()))
        cluster_scores = np.array(cluster_scores).astype(int)
        cluster_scores = cluster_scores[cluster_scores[:, 1].argsort()][::-1]

        # step: print a few top poisonous clusters
        for clust_rank, (clust_id, clust_score) in enumerate(cluster_scores.tolist()[:10]):
            logger.info(f'==> top poisoned clusters : rank {clust_rank:3d} cluster_id {clust_id:3d} score {clust_score}')

        logger.info(f'==> processed count : {processed_count:6d}/{len(train_val_dataset)} ({processed_count*100/len(train_val_dataset):.1f})')

        if args.prune_clusters:
            # step: remove a few least poisonous clusters
            rem = int(args.remove_per_iteration * len(candidate_clusters))
            candidate_clusters = cluster_scores[:-rem, 0].tolist()

        cur_iter += 1

    # step: save the poison scores or load them from the cache
    if use_cached_poison_scores:
        poison_scores = np.load(poison_scores_file)
    else:
        np.save(poison_scores_file, poison_scores)

    ######################################################################################################

    # step: get a few top poisonous images
    save_inds = poison_scores.argsort()[::-1][:100]

    inp, inp_titles = [], []
    for i in save_inds:
       inp.append(train_val_dataset[i][0])
       class_name = class_id_to_name[train_y[i, 0]]
       class_name = class_name if ',' not in class_name else class_name.split(',')[0]
       class_name = class_name.lower()
       inp_titles.append(class_name)
    inp = torch.stack(inp, dim=0)

    # step: save the top images and patches
    cam_images, out = run_gradcam(args.arch, model, inp)
    windows = extract_max_window(cam_images, inp, args.window_w)
    os.makedirs(os.path.join(args.save, 'all_top_poison_patches'), exist_ok=True)
    for i, win in enumerate(windows):
        win = denormalize(win)
        win = (win * 255).clamp(0, 255).numpy().astype(np.uint8)
        win = Image.fromarray(win)
        win.save(os.path.join(args.save, 'all_top_poison_patches', f'{i:05d}.png'))

    sorted_inds = poison_scores.argsort()[::-1]
    topks = [5, 10, 20, 50, 100, 500]
    accs = [train_p[sorted_inds[:k]].sum() * 100.0 / k for k in topks]
    logger.info('==> acc in top-k | ' + ' '.join(f'{k:7d}' for k in topks))
    logger.info('==> acc in top-k | ' + ' '.join(f'{acc:7.1f}' for acc in accs))


if __name__ == '__main__':
    main()
