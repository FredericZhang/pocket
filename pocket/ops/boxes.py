"""
Bounding box operations

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
from torchvision.ops import boxes

Tensor = torch.Tensor

def box_iou(boxes_1: Tensor, boxes_2: Tensor, encoding: str = 'coord') -> Tensor:
    """
    Compute intersection over union between boxes

    Arguments:
        boxes_1: (N, 4) Bounding boxes formatted as [[x1, y1, x2, y2],...]
        boxes_2: (M, 4) Bounding boxes formatted as [[x1, y1, x2, y2],...]
        encoding: A string that indicates what the boxes encode
            'coord': Coordinates of the two corners
            'pixel': Pixel indices of the two corners
    """
    if encoding == 'coord':
        return boxes.box_iou(boxes_1, boxes_2)
    elif encoding == 'pixel':
        w1 = (boxes_1[:, 2] - boxes_1[:, 0] + 1).clamp(min=0)
        h1 = (boxes_1[:, 3] - boxes_1[:, 1] + 1).clamp(min=0)
        s1 = w1 * h1
        w2 = (boxes_2[:, 2] - boxes_2[:, 0] + 1).clamp(min=0)
        h2 = (boxes_2[:, 3] - boxes_2[:, 1] + 1).clamp(min=0)
        s2 = w2 * h2

        n1 = len(boxes_1); n2 = len(boxes_2)
        i, j = torch.meshgrid(
            torch.arange(n1),
            torch.arange(n2)
        )
        i = i.flatten(); j = j.flatten()
        
        x1, y1 = torch.max(boxes_1[i, :2], boxes_2[j, :2]).unbind(1)
        x2, y2 = torch.min(boxes_1[i, 2:], boxes_2[j, 2:]).unbind(1)
        w_intr = (x2 - x1 + 1).clamp(min=0)
        h_intr = (y2 - y1 + 1).clamp(min=0)
        s_intr = w_intr * h_intr

        iou = s_intr / (s1[i] + s2[j] - s_intr)
        return iou.reshape(n1, n2)
    else:
        raise ValueError("The encoding type should be either \"coord\" or \"pixel\"")

def box_giou(boxes_1: Tensor, boxes_2: Tensor) -> Tensor:
    """
    Compute generalised intersection over union between boxes

    Reference: Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    Parameters:
    -----------
    boxes_1: torch.Tensor
        Bounding boxes of size (N, 4) formatted as [[x1, y1, x2, y2],...]
    boxes_2: torch.Tensor
        Bounding boxes of size (M, 4) formatted as [[x1, y1, x2, y2],...]

    Returns:
    --------
    giou: torch.Tensor
        Generalised IoUs of size (N, M)
    """
    i, j = torch.meshgrid(
        torch.arange(len(boxes_1)),
        torch.arange(len(boxes_2))
    )
    boxes_1 = boxes_1[i]
    boxes_2 = boxes_2[j]

    x1min, y1min, x2min, y2min = torch.min(boxes_1, boxes_2).unbind(-1)
    x1max, y1max, x2max, y2max = torch.max(boxes_1, boxes_2).unbind(-1)

    intersection = (x2min - x1max).clamp(min=0) * (y2min - y1max).clamp(min=0)
    convex_hull = (x2max - x1min) * (y2max - y1min)
    union = torch.prod(
        boxes_1[..., 2:] - boxes_1[..., :2], dim=-1
    ) + torch.prod(
        boxes_2[..., 2:] - boxes_2[..., :2], dim=-1
    ) - intersection

    iou = intersection / union
    giou = iou - (convex_hull - union) / convex_hull

    return giou