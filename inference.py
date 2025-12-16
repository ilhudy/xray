"""
============================================
🔮 X-Ray Segmentation 추론 스크립트
============================================
학습된 모델로 테스트 데이터 추론 후 CSV 파일 생성

사용법:
    # Config 파일 사용
    python inference.py --config configs/exp01_fcn_resnet50.yaml
    
    # 직접 모델 경로 지정
    python inference.py --model saved_models/exp01/best_model.pt --output output.csv
    
    # 모델 이름과 경로 직접 지정
    python inference.py --model saved_models/exp01/best_model.pt \
                        --model_name fcn_resnet50 \
                        --output my_submission.csv
"""

import argparse
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import XRayInferenceDataset
from src.models import get_model
from src.utils import encode_mask_to_rle, set_seed

# 기본 클래스 목록
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

IND2CLASS = {i: v for i, v in enumerate(CLASSES)}


def inference(
    model_path: str,
    test_image_root: str,
    output_csv: str,
    model_name: str = "fcn_resnet50",
    image_size: int = 512,
    original_size: int = 2048,
    batch_size: int = 2,
    threshold: float = 0.5,
    device: str = None,
):
    """
    추론 실행 및 CSV 생성
    
    Args:
        model_path: 학습된 모델 파일 경로 (.pt)
        test_image_root: 테스트 이미지 폴더 경로
        output_csv: 출력 CSV 파일 경로
        model_name: 모델 이름 (fcn_resnet50, deeplabv3_resnet101 등)
        image_size: 입력 이미지 크기
        original_size: 원본 이미지 크기 (출력 크기)
        batch_size: 배치 크기
        threshold: 이진화 임계값
        device: 디바이스 (None이면 자동 선택)
    """
    
    # 디바이스 설정
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    # Transform
    transform = A.Resize(image_size, image_size)
    
    # Dataset & DataLoader
    test_dataset = XRayInferenceDataset(
        image_root=test_image_root,
        transforms=transform,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    
    # 모델 로드
    print(f"📂 Loading model from {model_path}")
    model = get_model(model_name, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # 추론
    rles = []
    filename_and_class = []
    
    print("\n🔮 Starting inference...")
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device)
            outputs = model(images)["out"]
            
            # 원본 크기로 복원
            outputs = F.interpolate(
                outputs, size=(original_size, original_size), mode="bilinear"
            )
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > threshold).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
    
    # CSV 생성
    classes_list, filename_list = zip(
        *[x.split("_", 1) for x in filename_and_class]
    )
    image_names = [os.path.basename(f) for f in filename_list]
    
    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes_list,
        "rle": rles,
    })
    
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"🎉 Inference Complete!")
    print(f"{'='*60}")
    print(f"📁 Output CSV: {output_csv}")
    print(f"📊 Total predictions: {len(df)}")
    print(f"🖼️ Total images: {len(df) // len(CLASSES)}")
    print(f"{'='*60}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="X-Ray Segmentation Inference & CSV Generation"
    )
    
    # Config 파일 사용
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config yaml file path (선택사항)",
    )
    
    # 직접 지정
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="학습된 모델 파일 경로 (.pt)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="fcn_resnet50",
        help="모델 이름 (fcn_resnet50, deeplabv3_resnet101 등)",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default="data/test/DCM",
        help="테스트 이미지 폴더 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="출력 CSV 파일 경로",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="입력 이미지 크기",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="이진화 임계값",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="배치 크기",
    )
    
    args = parser.parse_args()
    
    # Config 파일 사용
    if args.config:
        print(f"📄 Loading config from {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        model_path = os.path.join(
            config["save"]["dir"], 
            config["save"]["model_name"]
        )
        
        inference(
            model_path=model_path,
            test_image_root=config["data"]["test_image_root"],
            output_csv=config["inference"]["output_csv"],
            model_name=config["model"]["name"],
            image_size=config["image"]["size"],
            original_size=config["image"]["original_size"],
            batch_size=config["inference"]["batch_size"],
            threshold=config["inference"]["threshold"],
        )
    
    # 직접 지정
    elif args.model:
        inference(
            model_path=args.model,
            test_image_root=args.test_root,
            output_csv=args.output,
            model_name=args.model_name,
            image_size=args.image_size,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
    
    else:
        print("❌ Error: --config 또는 --model 중 하나를 지정해주세요.")
        print("\n사용 예시:")
        print("  python inference.py --config configs/exp01_fcn_resnet50.yaml")
        print("  python inference.py --model saved_models/exp01/best_model.pt --output output.csv")
        parser.print_help()


if __name__ == "__main__":
    main()

