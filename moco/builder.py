# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from copy import deepcopy

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

class RMoCov1(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(RMoCov1, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.proj_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), deepcopy(self.encoder_q.fc))
            self.proj_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), deepcopy(self.encoder_k.fc))
            self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-1])
            self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-1])
            #self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            #self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)


        #assign encoder_q's weights to encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_pro_q, param_pro_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_pro_k.data.copy_(param_pro_q.data)
            param_pro_k.requires_grad = False
        self.rproj_q = deepcopy(self.proj_q)
        self.rproj_k = deepcopy(self.proj_k)
        #for rparam_q, rparam_k in zip(self.rproj_q.parameters(), self.rproj_k.parameters()):
            #rparam_k.data.copy_(rparam_q.data)
            #rparam_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0) #L2_norm is applied to dim channel
        self.register_buffer("queue_r", torch.randn(dim, K))
        self.queue_r = nn.functional.normalize(self.queue_r, dim=0) #L2_norm is applied to dim channel

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_angle", torch.zeros(K, dtype=torch.float64))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_pro_q, param_pro_k, rparam_pro_q, rparam_pro_k in zip(self.proj_q.parameters(),  self.proj_k.parameters(), self.rproj_q.parameters(), self.rproj_k.parameters()):
            param_pro_k.data = param_pro_k.data * self.m + param_pro_q.data * (1. - self.m)
            rparam_pro_k.data = rparam_pro_k.data * self.m + rparam_pro_q.data * (1. - self.m)
        

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_r, keys_r_angle):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys_r = concat_all_gather(keys_r)
        keys_r_angle = concat_all_gather(keys_r_angle)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_r[:, ptr:ptr + batch_size] = keys_r.T
        self.queue_angle[ptr:ptr + batch_size] = keys_r_angle
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _get_pos_neg_rotation(self, ang, threshold_pos, threshold_neg):
        ang = ang.reshape(-1,1)   #[N,1]
        match = torch.abs(ang - self.queue_angle.reshape(1,-1).repeat(ang.shape[0],1)) #[N, 65536]
        #select pos
        match_min_pos, match_idx_pos = torch.min(match, dim=1)
        keep_pos = torch.where(match_min_pos <= threshold_pos, True, False)
        #select neg
        bool_keep = torch.where(match <= threshold_neg, True, False)  #[N, 65536]
        bool_keep_inc = torch.sum(bool_keep, dim=0)
        keep_neg = torch.where(bool_keep_inc<1, True, False)
        return (self.queue_r[:,match_idx_pos], match_idx_pos, keep_pos),(self.queue_r[:,keep_neg], keep_neg)


    def forward(self, im_q, im_k, ang_q, ang_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        enc_q = self.encoder_q(im_q)  # queries: NxC
        enc_q = enc_q.squeeze()
        q = self.proj_q(enc_q)
        r_q = self.rproj_q(enc_q)
        q = nn.functional.normalize(q, dim=1) #apply l2norm on dim channel
        r_q = nn.functional.normalize(r_q, dim=1) #apply l2norm on dim channel

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            enc_k = self.encoder_k(im_k)  # keys: NxC
            enc_k = enc_k.squeeze()
            k = self.proj_k(enc_k)
            r_k = self.rproj_k(enc_k)
            k = nn.functional.normalize(k, dim=1)
            r_k = nn.functional.normalize(r_k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            r_k = self._batch_unshuffle_ddp(r_k, idx_unshuffle)

        # select pos and neg for angle branch
        r_pos, r_neg = self._get_pos_neg_rotation(ang_q, threshold_pos=1.0, threshold_neg=2.0)
        r_pos_k, r_pos_idx, r_pos_keep = r_pos
        r_neg_k, r_neg_keep = r_neg
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #inner-product
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # labels and logits for rotation
        #if r_pos_k.shape[1]==0 or r_neg_k.shape[1]==0:  #no pos or no neg, return loss_for_roration=0
        #    pass
        #else:
        
        if True not in r_pos_keep:
            r_l_pos = torch.einsum('nc,nc->n', [r_q, r_k]).unsqueeze(-1)
            r_l_neg = torch.einsum('nc,ck->nk', [r_q, r_neg_k.clone().detach()])
        else:
            # positive logits for roration: N2x1, where N2 <= N
            r_l_pos = torch.einsum('nc,nc->n', [r_q[r_pos_keep], r_pos_k[:,r_pos_keep].permute(1,0)]).unsqueeze(-1)
            # negative logits for roration: N2xK, where N2 <= 65536
            r_l_neg = torch.einsum('nc,ck->nk', [r_q[r_pos_keep], r_neg_k.clone().detach()])
        # logits
        r_logits = torch.cat([r_l_pos, r_l_neg], dim=1)
        r_logits /= self.T
        r_labels = torch.zeros(r_logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, r_k, ang_k)

        return logits, labels, r_logits, r_labels


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
