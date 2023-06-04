# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import time
import torch
import datetime
import torch.distributed as dist
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor
from collections import defaultdict, deque
from util.dist import get_world_size, is_dist_avail_and_initialized


class MetricLogger(object):
    def __init__(self, print_freq, delimiter="\t", vis=None, debug=False):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.vis = vis
        self.print_freq = print_freq
        self.debug = debug

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, epoch=None, header=None, batch_size=1):
        i = 0
        if header is None:
            header = 'Epoch: [{}]'.format(epoch)

        world_len_iterable = get_world_size() * len(iterable)

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(world_len_iterable))) + '.1f'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data_time: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i / batch_size, int(world_len_iterable / (get_world_size()*batch_size)), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i * get_world_size(), world_len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

                if self.vis is not None:
                    y_data = [self.meters[legend_name].median
                              for legend_name in self.vis.viz_opts['legend']
                              if legend_name in self.meters]
                    y_data.append(iter_time.median)

                    self.vis.plot(y_data, i * get_world_size() + (epoch - 1) * world_len_iterable)

                # DEBUG
                # if i != 0 and i % self.print_freq == 0:
                if self.debug and i % self.print_freq == 0:
                    break

            i += 1
            end = time.time()

        # if self.vis is not None:
        #     self.vis.reset()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def select_frame(self, fid):
        return NestedTensor(self.tensors[fid: fid+1], self.mask[fid: fid+1])

    def __repr__(self):
        return str(self.tensors)
    
    @classmethod
    def nested_batch_frame_list(cls, batch_vids):
        vid_lens = [len(batch_vids[i][-1]) for i in range(len(batch_vids))]
        max_len =  max(vid_lens)
        
        for i, vid_data in enumerate(batch_vids):
            vid, gt, vid_frames = vid_data
            vid_len = vid_lens[i]
            new_frames = torch.cat([vid_frames, torch.zeros(max_len-vid_len)-1])
            batch_vids[i] = (vid, gt, vid_len, new_frames)
        return cls(batch_vids)

    @classmethod
    def from_tensor_list(cls, tensor_list, do_round=False):
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = _max_by_axis([list(img.shape) for img in tensor_list])
         
            #import pdb;pdb.set_trace()
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        elif tensor_list[0].ndim == 4:  # videos
            max_size = tuple(max(s) for s in zip(*[clip.shape for clip in tensor_list]))  
            batch_shape = (len(tensor_list),) + max_size
            b, c, t, h, w = batch_shape

            assert b==1  # now batchsize is fixed as 1
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, t, h, w
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device

            nb_images = sum(
                clip.shape[1] for clip in tensor_list
            )  # total number of frames in the batch
            tensor = torch.zeros((nb_images, c, h, w), dtype=dtype, device=device)
            mask = torch.ones((nb_images, h, w), dtype=torch.bool, device=device)
            cur_dur = 0
            for i_clip, clip in enumerate(tensor_list):
                tensor[
                    cur_dur : cur_dur + clip.shape[1],
                    : clip.shape[0],
                    : clip.shape[2],
                    : clip.shape[3],
                ].copy_(clip.transpose(0, 1))
                mask[
                    cur_dur : cur_dur + clip.shape[1], : clip.shape[2], : clip.shape[3]
                ] = False
                cur_dur += clip.shape[1]
            #tensor, mask = tensor.reshape(b,t,c,h,w), mask.reshape(b,t,h,w) # uncomment it when batchsize>1
            tensor, mask = tensor.reshape(t,c,h,w), mask.reshape(t,h,w)
        else:
            raise ValueError("not supported")
        
        return cls(tensor, mask)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

    
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    elif tensor_list[0].ndim == 4:
        max_size = _max_by_axis([list(imgs.shape) for imgs in tensor_list])
        
        batch_shape = [len(tensor_list)] + max_size
        b, t, _, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, t, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:, :img.shape[1], :img.shape[2], :img.shape[3]].copy_(img)
            m[:, :img.shape[2], :img.shape[3]] = False
        
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def collate_fn(batch):
    batch = list(zip(*batch))  # [sample, target]
    batch[0] = NestedTensor.from_tensor_list(batch[0])  # sample
    return tuple(batch)


def collate_fn_val(batch):
    batch = NestedTensor.nested_batch_frame_list(batch)
    return batch


def target_to_cuda(target):
    _target = {}
    for k, v in target.items():     
        if isinstance(v, int) or isinstance(v, dict) or isinstance(v, str) or k=="groundtruth":
            _target[k] = v
        #elif k in ["verb_labels", "raw_verb_labels"]:
        elif k == "raw_verb_labels":
            _target[k] = [torch.as_tensor(verb).cuda(non_blocking=True) for verb in v]
        else:
            _target[k] = v.cuda(non_blocking=True)

    return _target


def sigmoid_focal_loss(inputs, targets, num_trajs, alpha:float=0.25, 
                       gamma:float=2, query_mask=None, reduction=True):

    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)


    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    if not reduction:
        return loss

    if query_mask is not None:
        loss = torch.stack([l[m].mean(0) for l, m in zip(loss, query_mask)])
        return loss.sum() / num_trajs
    
    return loss.mean(1).sum() / num_trajs

