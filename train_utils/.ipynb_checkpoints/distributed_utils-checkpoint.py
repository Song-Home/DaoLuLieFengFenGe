from collections import defaultdict, deque
import datetime
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist

import errno
import os

from .dice_coefficient_loss import multiclass_dice_coeff, build_target


class SmoothedValue(object):                                                          # 其主要功能是跟踪一系列值并提供对这些值的平滑处理
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):                                                     # 用于更新队列中的值，更新total(总和)和count(计数)
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
    def median(self):                                           # 返回队列中所有值的中位数
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()                                  # 返回队列中所有值的平均值

    @property
    def global_avg(self):
        return self.total / self.count                          # 返回自实例化以来所有值的全局平均值

    @property
    def max(self):                                              # 返回队列中的最大值
        return max(self.deque)

    @property                                                   # 返回队列中的最后一个值
    def value(self):
        return self.deque[-1]

    def __str__(self):                                          # 定义了类实例的字符串表示，包含中位数、平均数、全局平均数、最大值和当前值
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes):   
        self.num_classes = num_classes                                                 # 设置类别的数量
        self.mat = None                                                                # 初始化混淆矩阵为None

    def update(self, a, b):
        n = self.num_classes                                                           # n为类别数量
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)         # 创建混淆矩阵，形状为(n,n)，初始化为0
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)                                                     # 这一行创建了一个布尔张量k，其形状与输入张量a相同，用于确定哪些元素的值位于0到n-1之间，即真实标签在有效范围内。相当于找到了ingore index不为255的位置
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]                                     # 这里形成了4个索引，3，2，1，0
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)             # torch.bincount函数统计了inds中每个索引出现的次数，并将结果加到混淆矩阵self.mat上

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()                                                           # 重置混淆矩阵为全0

    def compute(self):
        h = self.mat.float()                                                           # 这一行将混淆矩阵的数据类型转换为浮点型，以便进行后续的数值计算
        # 计算全局预测准确率                                                             # torch.diag(h)返回的是一个包含混淆矩阵h对角线元素的一维张量，这意味着它是一个一维张量。
        acc_global = torch.diag(h).sum() / h.sum()                                     # 通过取混淆矩阵的对角线元素（即正确预测的样本数），然后将它们相加;最后除以混淆矩阵的总样本数（即所有预测的样本数）来得到准确率，这样得到整体数据集的平均准确率。
        # 计算每个类别的准确率                                                           # h.sum(1)返回的是一个包含每一行求和后的结果的一维张量
        acc = torch.diag(h) / h.sum(1)                                                 # 通过取混淆矩阵的对角线元素（即每个类别正确预测的样本数），然后将它们除以每行的总样本数（即每个类别的总样本数）来得到每个类别的准确率
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))                     # 计算每个类别的IoU。通过取混淆矩阵的对角线元素，然后将它们除以每行的总样本数加上每列的样本数，再减去对角线元素以避免重复计算。
        # 计算每个类别的召回率
        recall = torch.diag(h) / h.sum(1)
        # 计算每个类别的准确率
        precision = torch.diag(h) / h.sum(0)
        # 计算每个类别的F1分数
        f1 = 2 * (precision * recall) / (precision + recall)
        f1[torch.isnan(f1)] = 0  # 处理那些分母为0的情况，导致结果为NaN的情况

        return acc_global, acc, iu, recall, precision, f1                              # 这一行返回了计算得到的全局预测准确率、每个类别的准确率和每个类别的IoU.
# acc_global是一个标量张量，因为它是所有预测样本准确率的平均值，所以它是单个的浮点数
# acc是一个一维张量，其中每个元素对应一个类别的准确率
# iu是一个一维张量，其中每个元素对应一个类别的IoU
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)                                         # 在所有进程中合并混淆矩阵

    def __str__(self):
        acc_global, acc, iu, recall, precision, f1= self.compute()
        return (
            'global accuracy: {:.1f}\n'                                           # 全局准确率
            'each accuracy: {}\n'                                                 # 各个准确率
            'IoU: {}\n'                                                           # IoU
            'mean IoU: {:.1f}\n'
            'Recall: {:.1f}\n'
            'Precision: {:.1f}\n'
            'F1-score:{:.1f}'
            ).format(                                           # mean IoU
                acc_global.item() * 100,                                          # acc_global.item()用于提取张量中的单个数值
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],               # 将每个类别的准确率乘100，并将其格式化为一位小数的百分数字符串，并将字符串放入一个列表中
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],                # 将每个类别的IoU乘100，并将其格式化为一位小数的百分数字符串，并将字符串放入一个列表中
                iu.mean().item() * 100,                                           # 求每个类别的的IoU平均值乘100
                recall[1].item() * 100,
                precision[1].item() * 100,
                f1[1].item() * 100 
                )                                           


class DiceCoefficient(object):
    def __init__(self, num_classes: int = 2, ignore_index: int = -100): # 初始化cumulative_dice(用于累计Dice系数的变量)，num_classes和ignore_index，计数器count
        self.cumulative_dice = None
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.count = None

    def update(self, pred, target):
        if self.cumulative_dice is None: # 这个方法用于更新Dice系数。它首先检查cumulative_dice和count是否为None，如果是，则分别初始化为零值的张量
            self.cumulative_dice = torch.zeros(1, dtype=pred.dtype, device=pred.device) 
        if self.count is None:
            self.count = torch.zeros(1, dtype=pred.dtype, device=pred.device) # 创建一个与pred张量数据类型和设备都匹配的、大小为1且所有元素都为0的张量
        # compute the Dice score, ignoring background
        pred = F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, 3, 1, 2).float() # 它将预测结果转换为one-hot编码
        dice_target = build_target(target, self.num_classes, self.ignore_index)            # 并调用build_target函数和multiclass_dice_coeff函数来计算Dice系数
        self.cumulative_dice += multiclass_dice_coeff(pred[:, 1:], dice_target[:, 1:], ignore_index=self.ignore_index) # 更新累积的Dice系数和计数器
        self.count += 1

    @property
    def value(self):                                             # 用于计算平均Dice系数
        if self.count == 0:
            return 0
        else:
            return self.cumulative_dice / self.count

    def reset(self):                                             # 这个方法用于重置cumulative_dice和count
        if self.cumulative_dice is not None:
            self.cumulative_dice.zero_()

        if self.count is not None:
            self.count.zeros_()

    def reduce_from_all_processes(self):                         # 这个方法用于在分布式环境中从所有进程收集数据
        if not torch.distributed.is_available():                 # 它首先检查Pytorch分布式是否可用
            return
        if not torch.distributed.is_initialized():               # 它检查Pytorch分布式是否已初始化
            return 
        torch.distributed.barrier()                              # 同步所有进程的数据
        torch.distributed.all_reduce(self.cumulative_dice)       # 减少来自所有进程的数据
        torch.distributed.all_reduce(self.count)


class MetricLogger(object):                                                 # 定义了一个名为MetricLogger的类，继承自object
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
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
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
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
