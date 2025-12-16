"""
============================================
🏗️ 모델 정의
============================================
다양한 Segmentation 모델을 정의하고 불러오는 모듈
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    모델을 생성하여 반환
    
    Args:
        model_name (str): 모델 이름
            - "fcn_resnet50": FCN with ResNet50 backbone
            - "fcn_resnet101": FCN with ResNet101 backbone
            - "deeplabv3_resnet50": DeepLabV3 with ResNet50 backbone
            - "deeplabv3_resnet101": DeepLabV3 with ResNet101 backbone
            - "deeplabv3_mobilenet": DeepLabV3 with MobileNetV3 backbone
        num_classes (int): 출력 클래스 수
        pretrained (bool): 사전학습 가중치 사용 여부
    
    Returns:
        nn.Module: 생성된 모델
    
    Example:
        >>> model = get_model("fcn_resnet50", num_classes=29)
    """
    
    model_name = model_name.lower()
    
    if model_name == "fcn_resnet50":
        model = models.segmentation.fcn_resnet50(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    elif model_name == "fcn_resnet101":
        model = models.segmentation.fcn_resnet101(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    elif model_name == "deeplabv3_resnet50":
        model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    elif model_name == "deeplabv3_resnet101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    elif model_name == "deeplabv3_mobilenet":
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    else:
        raise ValueError(
            f"지원하지 않는 모델입니다: {model_name}\n"
            f"지원 모델: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, "
            f"deeplabv3_resnet101, deeplabv3_mobilenet"
        )
    
    return model


# ============================================
# 📌 Custom 모델 예시 (필요시 추가)
# ============================================

class SimpleUNet(nn.Module):
    """
    간단한 U-Net 구현 예시
    
    실험용으로 사용하거나 참고용으로 작성
    실제 사용시 segmentation-models-pytorch 라이브러리 추천
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 29):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # torchvision 모델과 동일한 출력 형식
        return {"out": self.out(d1)}


# 모델 등록 (get_model에서 사용)
CUSTOM_MODELS = {
    "simple_unet": SimpleUNet,
}


def get_custom_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """커스텀 모델 생성"""
    if model_name not in CUSTOM_MODELS:
        raise ValueError(f"Unknown custom model: {model_name}")
    return CUSTOM_MODELS[model_name](num_classes=num_classes, **kwargs)

