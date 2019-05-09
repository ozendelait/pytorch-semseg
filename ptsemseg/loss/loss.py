import torch
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True, try_downsampling=False):
    if isinstance(input, tuple):
        input = input[0]
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    if try_downsampling and h < ht and w < wt: #downsample labels
        # cannot use F.interpolate(target.view(1,nt,ht,wt), size=(h, w), mode="nearest") due to
        # _thnn_upsample_nearest2d_forward is not implemented for type torch.cuda.LongTensor at
        scale0 = int(ht/h+0.5) #try to downsample using slicing
        if abs(wt//scale0-w) <2:
            target = target[:,::scale0,::scale0].contiguous()
            nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, ignore_index=250, reduction=['sum','mean'][size_average]
    )
    return loss


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None, try_downsampling=False):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average, try_downsampling=try_downsampling)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            target.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        target_scaled = target
        n, c, h, w = inp.size()
        nt, ht, wt = target.size()
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target_scaled, weight=weight, size_average=size_average, try_downsampling=try_downsampling
        )

    return loss


def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, reduce=False, size_average=False, ignore_index=250
        )

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)
