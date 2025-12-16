"""
============================================
🚀 X-Ray Segmentation 메인 실행 코드
============================================
학습 및 추론을 실행하는 메인 스크립트

사용법:
    # 학습
    python main.py --config configs/exp01_fcn_resnet50.yaml --mode train
    
    # 추론
    python main.py --config configs/exp01_fcn_resnet50.yaml --mode inference
    
    # 학습 + 추론
    python main.py --config configs/exp01_fcn_resnet50.yaml --mode all
"""

import argparse
import datetime
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import XRayDataset, XRayInferenceDataset
from src.losses import get_loss
from src.models import get_model
from src.utils import (
    dice_coef,
    encode_mask_to_rle,
    get_device,
    print_model_info,
    save_model,
    set_seed,
)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"📄 Config loaded from {config_path}")
    return config


def get_transforms(config: dict, is_train: bool = True):
    """Augmentation 변환 생성"""
    aug_config = config["augmentation"]["train" if is_train else "valid"]
    
    transforms_list = []
    
    # Resize (필수)
    transforms_list.append(A.Resize(aug_config["resize"], aug_config["resize"]))
    
    # 학습시 추가 augmentation
    if is_train:
        if aug_config.get("horizontal_flip", False):
            transforms_list.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get("vertical_flip", False):
            transforms_list.append(A.VerticalFlip(p=0.5))
        
        if aug_config.get("rotate", False):
            limit = aug_config.get("rotate_limit", 15)
            transforms_list.append(A.Rotate(limit=limit, p=0.5))
        
        if aug_config.get("brightness_contrast", False):
            transforms_list.append(A.RandomBrightnessContrast(p=0.3))
        
        if aug_config.get("elastic_transform", False):
            transforms_list.append(A.ElasticTransform(p=0.3))
        
        if aug_config.get("grid_distortion", False):
            transforms_list.append(A.GridDistortion(p=0.3))
    
    return A.Compose(transforms_list)


def get_optimizer(model, config: dict):
    """옵티마이저 생성"""
    opt_name = config["training"].get("optimizer", "adam").lower()
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 1e-6)
    
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, config: dict):
    """스케줄러 생성"""
    scheduler_name = config["training"].get("scheduler")
    
    if scheduler_name is None:
        return None
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["epochs"]
        )
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def validation(epoch, model, data_loader, criterion, classes, device, thr=0.5):
    """검증 수행"""
    print(f"\n🔍 Validation #{epoch}")
    model.eval()
    
    dices = []
    total_loss = 0
    cnt = 0
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)["out"]
            
            # 크기 맞추기
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).float()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice.cpu())
    
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    
    # 클래스별 Dice 출력
    print("\n📊 Class-wise Dice Scores:")
    for c, d in zip(classes, dices_per_class):
        print(f"  {c:<15}: {d.item():.4f}")
    
    avg_dice = torch.mean(dices_per_class).item()
    avg_loss = total_loss / cnt
    
    print(f"\n📈 Average Dice: {avg_dice:.4f}")
    print(f"📉 Average Loss: {avg_loss:.4f}")
    
    return avg_dice


def train(config: dict, device: str):
    """학습 실행"""
    print("\n" + "=" * 60)
    print("🎓 TRAINING START")
    print("=" * 60)
    
    # 설정 추출
    classes = config["classes"]
    num_classes = len(classes)
    
    # Transform 생성
    train_transform = get_transforms(config, is_train=True)
    valid_transform = get_transforms(config, is_train=False)
    
    # Dataset 생성
    train_dataset = XRayDataset(
        image_root=config["data"]["train_image_root"],
        label_root=config["data"]["train_label_root"],
        classes=classes,
        is_train=True,
        transforms=train_transform,
        n_splits=config["training"]["n_splits"],
        fold=config["training"]["fold"],
    )
    
    valid_dataset = XRayDataset(
        image_root=config["data"]["train_image_root"],
        label_root=config["data"]["train_label_root"],
        classes=classes,
        is_train=False,
        transforms=valid_transform,
        n_splits=config["training"]["n_splits"],
        fold=config["training"]["fold"],
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    
    # 모델 생성
    model = get_model(
        config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
    )
    model = model.to(device)
    print_model_info(model, config["model"]["name"])
    
    # Loss, Optimizer, Scheduler 설정
    loss_config = config.get("loss", {})
    loss_name = loss_config.get("name", "bce")
    loss_kwargs = {k: v for k, v in loss_config.items() if k != "name"}
    criterion = get_loss(loss_name, **loss_kwargs)
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    print(f"\n📌 Loss: {loss_name}")
    print(f"📌 Optimizer: {config['training'].get('optimizer', 'adam')}")
    print(f"📌 Scheduler: {config['training'].get('scheduler', 'None')}")
    
    # 학습 루프
    best_dice = 0.0
    epochs = config["training"]["epochs"]
    val_every = config["training"]["val_every"]
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if scheduler is not None:
            scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"\n📊 Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
        
        # 검증
        if (epoch + 1) % val_every == 0:
            dice = validation(
                epoch + 1, model, valid_loader, criterion, classes, device
            )
            
            if dice > best_dice:
                print(f"\n🎉 Best Dice improved: {best_dice:.4f} → {dice:.4f}")
                best_dice = dice
                save_model(
                    model,
                    config["save"]["dir"],
                    config["save"]["model_name"],
                )
    
    print("\n" + "=" * 60)
    print(f"🏆 Training Complete! Best Dice: {best_dice:.4f}")
    print("=" * 60)


def inference(config: dict, device: str):
    """추론 실행"""
    print("\n" + "=" * 60)
    print("🔮 INFERENCE START")
    print("=" * 60)
    
    classes = config["classes"]
    num_classes = len(classes)
    ind2class = {i: v for i, v in enumerate(classes)}
    
    # Transform 생성
    test_transform = get_transforms(config, is_train=False)
    
    # Dataset 생성
    test_dataset = XRayInferenceDataset(
        image_root=config["data"]["test_image_root"],
        transforms=test_transform,
    )
    
    # DataLoader 생성
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    
    # 모델 로드
    model = get_model(
        config["model"]["name"],
        num_classes=num_classes,
        pretrained=False,
    )
    
    model_path = os.path.join(config["save"]["dir"], config["save"]["model_name"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"📂 Model loaded from {model_path}")
    
    # 추론
    rles = []
    filename_and_class = []
    thr = config["inference"]["threshold"]
    original_size = config["image"]["original_size"]
    
    with torch.no_grad():
        for images, image_names in tqdm(test_loader, desc="Inferencing"):
            images = images.to(device)
            outputs = model(images)["out"]
            
            outputs = F.interpolate(
                outputs, size=(original_size, original_size), mode="bilinear"
            )
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{ind2class[c]}_{image_name}")
    
    # CSV 저장
    classes_list, filename_list = zip(
        *[x.split("_", 1) for x in filename_and_class]
    )
    image_names = [os.path.basename(f) for f in filename_list]
    
    df = pd.DataFrame({
        "image_name": image_names,
        "class": classes_list,
        "rle": rles,
    })
    
    output_csv = config["inference"]["output_csv"]
    df.to_csv(output_csv, index=False)
    
    print(f"\n💾 Results saved to {output_csv}")
    print(f"📊 Total predictions: {len(df)}")
    
    print("\n" + "=" * 60)
    print("🏆 Inference Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="X-Ray Segmentation Training/Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "all"],
        default="train",
        help="Execution mode: train, inference, or all",
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 시드 설정
    set_seed(config.get("seed", 42))
    
    # 디바이스 설정
    device = get_device()
    
    # 실험 정보 출력
    print("\n" + "=" * 60)
    print("📋 EXPERIMENT INFO")
    print("=" * 60)
    print(f"Name: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Author: {config['experiment']['author']}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 실행
    if args.mode in ["train", "all"]:
        train(config, device)
    
    if args.mode in ["inference", "all"]:
        inference(config, device)


if __name__ == "__main__":
    main()

