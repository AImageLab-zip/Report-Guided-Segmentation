import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLoss(nn.Module):
    """
    Sigmoid contrastive loss (SigLIP-style) over a batch.

    Expects:
      img_emb: [B, D]
      txt_emb: [B, D]
    Returns:
      scalar loss

    Usage:
    criterion = SigLoss(mode = SigLoss.STANDARD)
    ...
    loss = criterion(img_emb, txt_emg)
    """
    STANDARD = 0        # Default SigLipLoss implementation
    ONLY_POSITIVES = 1  # Only positive terms contribute
    CONTINUOUS = 2      # Continuous weights in range [-1; +1] instead of hard -1 +1

    def __init__(
        self,
        temperature_init: float = 0.1,      
        bias_init: float = -10.0,             
        learnable_temperature: bool = True,
        learnable_bias: bool = True,
        mode = None
    ):
        super().__init__()

        init_t = float(torch.log(torch.tensor(1.0 / temperature_init))) #log(10)

        if learnable_temperature:
            self.t_prime = nn.Parameter(torch.tensor(init_t))
        else:
            self.register_buffer("t_prime", torch.tensor(init_t))

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias_init))
        else:
            self.register_buffer("bias", torch.tensor(bias_init))

        assert mode is not None, 'Please, pass an explicit mode param.'
        
    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        assert img_emb.dim() == 2 and txt_emb.dim() == 2, "Expected [B,D] embeddings"
        assert img_emb.shape == txt_emb.shape, f"Shape mismatch: {img_emb.shape} vs {txt_emb.shape}"

        # l2 normalization of embeddings
        img = F.normalize(img_emb, dim=-1)
        txt = F.normalize(txt_emb, dim=-1)

        # similarity matrix [B,B] -> diagonal entries = positive pairs, off-diagonal = negatives
        logits = img @ txt.t()

        # t = exp(t_prime)
        # logits = logits * t + bias
        if self.t_prime is not None:
            t = torch.exp(t_prime).clamp(max=100) 
            logits = logits * t
        if self.bias is not None:
            logits = logits + bias

        # labels: 2 * eye(B) - ones(B)
        B = img.shape[0]


        match self.mode:
            case self.STANDARD:
                labels = 2 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1  
                loss = -F.logsigmoid(labels * logits).sum() / B
            case self.ONLY_POSITIVES:
                labels = 2 * torch.eye(B, device=logits.device, dtype=logits.dtype) - 1  
                loss = -F.logsigmoid(labels * logits)
                loss = (torch.eye(B, device=logits.device, dtype=logits.dtype) * loss).sum() / B 
            case self.CONTINUOUS:
                labels = txt @ txt.t()
                loss = -F.logsigmoid(labels * logits).sum() / B
            case _:
                raise RuntimeError(f'Invalid mode:{self.mode}')



        #version 2
        # loss = -F.logsigmoid(labels * logits)
        # mask = (report_idx[:, None] != report_idx[None, :]).float()
        # mask += torch.eye(B, device=logits.device, dtype=logits.dtype)  # keep positives
        # masked_loss = (mask * loss).sum() / mask.sum().clamp(min=1.0)
        
        return loss



def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_image_features, image_features)
        torch.distributed.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class DistributedSigLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            continuous_weights = False
    ):
        super().__init__()
        self.cache_labels = False
        self.bidir = True
        self.continuous_weights = continuous_weights

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if rank == 0: 
                print(f'Running on {world_size} gpus')
        else:
            rank = 0
            world_size = 1
            print('Running on only one gpu')

        self.world_size = world_size
        self.rank = rank
    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, t_prime, bias=None):
        t = torch.exp(t_prime)
        logits = t * image_features @ text_features.T
        if bias is not None:
            logits += bias
        return logits

    def _loss(self, image_features, text_features, t_prime, bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, t_prime, bias)
        if self.continuous_weights:
            labels = text_features @ text_features.T
        else:
            labels = self.get_ground_truth(
                image_features.device,
                image_features.dtype,
                image_features.shape[0],
                negative_only=negative_only,
            )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, img, txt, t_prime, bias):

        image_features =  F.normalize(img, dim=-1)
        text_features = F.normalize(txt,dim=-1)
        loss = self._loss(image_features, text_features, t_prime, bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            t_prime,
                            bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        t_prime,
                        bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        t_prime,
                        bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return loss
