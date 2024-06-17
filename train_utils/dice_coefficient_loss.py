import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    # Dice loss (objective to minimize) between 0 and 1
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

def iou_loss(x: torch.Tensor, target: torch.Tensor,ignore_index: int = -100):
    x = nn.functional.softmax(x, dim=1)
    batch_size = x.shape[0]
    num_classes = x.shape[1]
    ious = []
    
    for i in range(batch_size):
        for c in range(num_classes):
            x_i = x[i, c].reshape(-1)  # 获取第i个样本和第c个类别的预测值，并展平
            t_i = target[i, c].reshape(-1)  # 获取对应的真实标签，并展平
            
            if ignore_index >= 0:
                valid_mask = t_i != ignore_index  # 创建一个有效区域的掩码
                x_i = x_i[valid_mask]  # 应用掩码，提取出感兴趣的预测值
                t_i = t_i[valid_mask]  # 应用掩码，提取出感兴趣的真实值
            
            # 计算交集
            intersection = torch.sum(x_i * t_i)
            # 计算并集
            union = torch.sum(x_i) + torch.sum(t_i) - intersection
            
            # 计算IoU，并避免除以0
            if union > 0:
                ious.append(intersection / union)
    
    # 计算平均IoU，如果ious列表为空（所有目标都是忽略索引），则返回0
    avg_iou = sum(ious) / len(ious) if ious else torch.tensor(0.0)
    
    # 返回1 - 平均IoU作为损失
    return 1 - avg_iou
