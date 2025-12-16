"""
============================================
🛠️ 유틸리티 함수
============================================
학습 및 추론에 필요한 유틸리티 함수 모음
"""

import os
import random
import numpy as np
import torch


# ============================================
# 🎲 시드 고정
# ============================================

def set_seed(seed: int = 42):
    """
    재현성을 위한 시드 고정
    
    Args:
        seed (int): 시드 값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🎲 Seed set to {seed}")


# ============================================
# 💾 모델 저장/불러오기
# ============================================

def save_model(model, save_dir: str, file_name: str = "best_model.pt"):
    """
    모델 저장
    
    Args:
        model: 저장할 모델
        save_dir (str): 저장 디렉토리
        file_name (str): 파일 이름
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)
    print(f"💾 Model saved to {output_path}")


def load_model(model, model_path: str, device: str = "cuda"):
    """
    모델 불러오기
    
    Args:
        model: 모델 인스턴스
        model_path (str): 가중치 파일 경로
        device (str): 디바이스
    
    Returns:
        불러온 모델
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"📂 Model loaded from {model_path}")
    return model


# ============================================
# 📊 평가 지표
# ============================================

def dice_coef(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Dice Coefficient 계산
    
    Args:
        y_true: 정답 마스크
        y_pred: 예측 마스크
        eps: 분모가 0이 되는 것을 방지
    
    Returns:
        클래스별 Dice coefficient (shape: [batch_size, num_classes])
    """
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    return (2.0 * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)


# ============================================
# 🔄 RLE 인코딩/디코딩
# ============================================

def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    마스크를 RLE(Run-Length Encoding)로 인코딩
    
    Args:
        mask: 이진 마스크 (numpy array)
    
    Returns:
        RLE 문자열
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def decode_rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """
    RLE를 마스크로 디코딩
    
    Args:
        rle: RLE 문자열
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        이진 마스크 (numpy array)
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)


# ============================================
# 🎨 시각화
# ============================================

# 29개 클래스용 색상 팔레트
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]


def label2rgb(label: np.ndarray, palette: list = None) -> np.ndarray:
    """
    라벨 마스크를 RGB 이미지로 변환
    
    Args:
        label: 라벨 마스크 (shape: [num_classes, H, W])
        palette: 색상 팔레트 (기본값: PALETTE)
    
    Returns:
        RGB 이미지 (shape: [H, W, 3])
    """
    if palette is None:
        palette = PALETTE
    
    image_size = label.shape[1:] + (3,)
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = palette[i % len(palette)]
    
    return image


# ============================================
# 📋 기타 유틸리티
# ============================================

def get_device():
    """사용 가능한 디바이스 반환"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    return device


def count_parameters(model) -> int:
    """모델의 학습 가능한 파라미터 수 반환"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, model_name: str = "Model"):
    """모델 정보 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"\n{'='*50}")
    print(f"📊 {model_name} Info")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"{'='*50}\n")

