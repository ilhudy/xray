"""
============================================
📉 Loss 함수 정의
============================================
Segmentation 학습을 위한 다양한 Loss 함수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss
    
    Segmentation에서 많이 사용되는 loss로, 
    예측과 정답의 겹치는 영역을 최대화
    
    Args:
        smooth (float): 분모가 0이 되는 것을 방지
    """
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss 조합
    
    BCE는 픽셀 단위 학습, Dice는 영역 단위 학습에 효과적
    두 loss를 조합하여 더 안정적인 학습 가능
    
    Args:
        bce_weight (float): BCE loss 가중치
        dice_weight (float): Dice loss 가중치
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss
    
    클래스 불균형 문제 해결에 효과적
    쉬운 샘플의 가중치를 줄이고 어려운 샘플에 집중
    
    Args:
        alpha (float): 양성 클래스 가중치
        gamma (float): focusing parameter (클수록 어려운 샘플에 집중)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss
    
    Dice Loss의 일반화 버전
    False Positive와 False Negative의 가중치를 조절 가능
    
    Args:
        alpha (float): False Positive 가중치
        beta (float): False Negative 가중치
        smooth (float): 분모가 0이 되는 것을 방지
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)
        
        tp = (pred_flat * target_flat).sum(-1)
        fp = (pred_flat * (1 - target_flat)).sum(-1)
        fn = ((1 - pred_flat) * target_flat).sum(-1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class CombinedLoss(nn.Module):
    """
    여러 Loss의 조합
    
    다양한 loss를 가중치와 함께 조합하여 사용
    
    Args:
        losses (list): (loss_fn, weight) 튜플의 리스트
    
    Example:
        >>> loss_fn = CombinedLoss([
        ...     (nn.BCEWithLogitsLoss(), 0.5),
        ...     (DiceLoss(), 0.3),
        ...     (FocalLoss(), 0.2),
        ... ])
    """
    
    def __init__(self, losses: list):
        super().__init__()
        self.losses = nn.ModuleList([loss_fn for loss_fn, _ in losses])
        self.weights = [weight for _, weight in losses]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss


# ============================================
# 📌 Loss 함수 선택 헬퍼
# ============================================

def get_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Loss 함수 이름으로 생성
    
    Args:
        loss_name (str): Loss 함수 이름
            - "bce": BCEWithLogitsLoss
            - "dice": DiceLoss
            - "bce_dice": BCEDiceLoss
            - "focal": FocalLoss
            - "tversky": TverskyLoss
        **kwargs: Loss 함수에 전달할 추가 인자
    
    Returns:
        nn.Module: Loss 함수
    
    Example:
        >>> loss_fn = get_loss("bce_dice", bce_weight=0.5, dice_weight=0.5)
    """
    
    loss_name = loss_name.lower()
    
    loss_dict = {
        "bce": nn.BCEWithLogitsLoss,
        "dice": DiceLoss,
        "bce_dice": BCEDiceLoss,
        "focal": FocalLoss,
        "tversky": TverskyLoss,
    }
    
    if loss_name not in loss_dict:
        raise ValueError(
            f"지원하지 않는 loss입니다: {loss_name}\n"
            f"지원 loss: {list(loss_dict.keys())}"
        )
    
    return loss_dict[loss_name](**kwargs)

