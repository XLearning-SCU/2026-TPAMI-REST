import math
import torch
import torch.nn as nn

def convert_weights_to_bf16(model: nn.Module):
    """Safely convert applicable model parameters to bfloat16"""

    def _convert_weights_to_bf16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if l.weight is not None:
                l.weight.data = l.weight.data.to(torch.bfloat16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.bfloat16)

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias", "bias_k", "bias_v",
            ]:
                tensor = getattr(l, attr, None)
                if isinstance(tensor, torch.Tensor):
                    tensor.data = tensor.data.to(torch.bfloat16)

        for name in ["text_projection", "proj"]:
            attr = getattr(l, name, None)
            if isinstance(attr, torch.Tensor):
                attr.data = attr.data.to(torch.bfloat16)
            elif isinstance(attr, nn.Module):
                if hasattr(attr, "weight") and isinstance(attr.weight, torch.Tensor):
                    attr.weight.data = attr.weight.data.to(torch.bfloat16)
                if hasattr(attr, "bias") and isinstance(attr.bias, torch.Tensor) and attr.bias is not None:
                    attr.bias.data = attr.bias.data.to(torch.bfloat16)

    model.apply(_convert_weights_to_bf16)

def convert_weights_to_fp16(model: nn.Module):
    """Safely convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            if l.weight is not None:
                l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias", "bias_k", "bias_v",
            ]:
                tensor = getattr(l, attr, None)
                if isinstance(tensor, torch.Tensor):
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            attr = getattr(l, name, None)
            if isinstance(attr, torch.Tensor):
                attr.data = attr.data.half()
            elif isinstance(attr, nn.Module):
                if hasattr(attr, "weight") and isinstance(attr.weight, torch.Tensor):
                    attr.weight.data = attr.weight.data.half()
                if hasattr(attr, "bias") and isinstance(attr.bias, torch.Tensor) and attr.bias is not None:
                    attr.bias.data = attr.bias.data.half()

    model.apply(_convert_weights_to_fp16)
    
def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
        
import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

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


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

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

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)        
        
@torch.no_grad()
def itm_eval(scores_v2a, scores_a2v):
    # Video->Audio 
    ranks = np.zeros(scores_v2a.shape[0])
    average_precisions_v2a = []
    for index, score in enumerate(scores_v2a):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        
        # The correct match is always the diagonal element (index = index)
        relevant_position = np.where(inds == index)[0][0]
        rank = relevant_position
        
        ranks[index] = rank

        # Calculate average precision
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate([relevant_position])]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_v2a.append(average_precision)

    # Compute metrics for v2a
    v2a_r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    v2a_r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    v2a_r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    # Audio->Video 
    ranks = np.zeros(scores_a2v.shape[0])
    average_precisions_a2v = []
    for index, score in enumerate(scores_a2v):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        
        # The correct match is always the diagonal element (index = index)
        relevant_position = np.where(inds == index)[0][0]
        rank = relevant_position
        
        ranks[index] = rank

        # Calculate average precision
        precisions = [(i + 1) / (pos + 1) for i, pos in enumerate([relevant_position])]
        average_precision = np.mean(precisions) if precisions else 0
        average_precisions_a2v.append(average_precision)

    # Compute metrics for a2v
    a2v_r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    a2v_r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    a2v_r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    # Calculate means
    v2a_r_mean = round((v2a_r1 + v2a_r5 + v2a_r10) / 3, 2)
    a2v_r_mean = round((a2v_r1 + a2v_r5 + a2v_r10) / 3, 2)
    r_mean = round((v2a_r_mean + a2v_r_mean) / 2, 2)

    # Calculate R_sum
    r_sum = v2a_r1 + v2a_r5 + v2a_r10 + a2v_r1 + a2v_r5 + a2v_r10

    # Calculate mAP and format to two decimal places
    mAP_v2a = round(np.mean(average_precisions_v2a) * 100, 2)
    mAP_a2v = round(np.mean(average_precisions_a2v) * 100, 2)

    eval_result =  {
        'v2a_r1': v2a_r1,
        'v2a_r5': v2a_r5,
        'v2a_r10': v2a_r10,
        'v2a_r_mean': v2a_r_mean,
        'v2a_map': mAP_v2a,
        'a2v_r1': a2v_r1,
        'a2v_r5': a2v_r5,
        'a2v_r10': a2v_r10,
        'a2v_r_mean': a2v_r_mean,
        'a2v_map': mAP_a2v,
        'r_mean': r_mean,
        'r_sum': r_sum
    }
    return eval_result
