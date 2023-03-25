import torch
import torch.nn as nn


def rand_bbox(size, lam):
    W, H = size
    cut_rat = (1. - lam).sqrt()
    cut_w = (W * cut_rat).to(torch.long)
    cut_h = (H * cut_rat).to(torch.long)

    cx = torch.zeros_like(cut_w, dtype=cut_w.dtype).random_(0, W)
    cy = torch.zeros_like(cut_h, dtype=cut_h.dtype).random_(0, H)

    bbx1 = (cx - cut_w // 2).clamp(0, W)
    bby1 = (cy - cut_h // 2).clamp(0, H)
    bbx2 = (cx + cut_w // 2).clamp(0, W)
    bby2 = (cy + cut_h // 2).clamp(0, H)

    new_lam = 1. - (bbx2 - bbx1).to(lam.dtype) * (bby2 - bby1).to(lam.dtype) / (W * H)

    return (bbx1, bby1, bbx2, bby2), new_lam


def cutmix(x, alpha):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]

    # create beta distribution and sample lambda from it
    beta = torch.distributions.beta.Beta(*alpha)
    lam = beta.sample().to(device=x.device)
    lam = torch.max(lam, 1. - lam)

    # sample a random bounding box
    (bbx1, bby1, bbx2, bby2), lam = rand_bbox(x.shape[-2:], lam)

    # sample random indices for cutmix
    randind = torch.randperm(x.shape[0], device=x.device)
    output = x.clone()
    output[..., bbx1:bbx2, bby1:bby2] = output[randind][..., bbx1:bbx2, bby1:bby2]

    return output, randind, lam


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, icutmix=False, alpha=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T
        self.icutmix = icutmix
        self.alpha = alpha

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def contrastive_icutmix_loss(self, q, labels_aux, lam, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        offset = N * torch.distributed.get_rank()
        labels = (torch.arange(N, dtype=torch.long) + offset).cuda()
        labels_aux = (labels_aux + offset).cuda()
        loss =  lam * nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
        loss = loss + (1. - lam) * nn.CrossEntropyLoss()(logits, labels_aux) * (2 * self.T)
        return loss

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        if self.icutmix:
            x1_mixed, y1_aux, lam1 = cutmix(x1, self.alpha)
            x2_mixed, y2_aux, lam2 = cutmix(x2, self.alpha)
            q1_mixed = self.predictor(self.base_encoder(x1_mixed))
            q2_mixed = self.predictor(self.base_encoder(x2_mixed))
        else:
            q1 = self.predictor(self.base_encoder(x1))
            q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        if self.icutmix:
            l1 = self.contrastive_icutmix_loss(q1_mixed, y1_aux, lam1, k2) 
            l2 = self.contrastive_icutmix_loss(q2_mixed, y2_aux, lam2, k1)
            return l1 + l2
        else:
            l1 = self.contrastive_loss(q1, k2)
            l2 = self.contrastive_loss(q2, k1)
            return l1 + l2


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
