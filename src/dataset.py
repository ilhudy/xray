"""
============================================
📊 Dataset 클래스 정의
============================================
X-Ray 이미지 segmentation을 위한 Custom Dataset 클래스
"""

import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
import torch


class XRayDataset(Dataset):
    """
    학습/검증용 X-Ray Dataset
    
    Args:
        image_root (str): 이미지 폴더 경로
        label_root (str): 라벨(JSON) 폴더 경로
        classes (list): 클래스 이름 리스트
        is_train (bool): 학습용 데이터셋 여부
        transforms: albumentations 변환
        n_splits (int): K-Fold 분할 수
        fold (int): 사용할 fold 번호 (0부터 시작)
    """
    
    def __init__(
        self,
        image_root: str,
        label_root: str,
        classes: list,
        is_train: bool = True,
        transforms=None,
        n_splits: int = 5,
        fold: int = 0,
    ):
        self.image_root = image_root
        self.label_root = label_root
        self.classes = classes
        self.class2ind = {v: i for i, v in enumerate(classes)}
        self.is_train = is_train
        self.transforms = transforms
        
        # 이미지 파일 목록 수집
        pngs = self._collect_files(image_root, ".png")
        jsons = self._collect_files(label_root, ".json")
        
        # 파일 정렬
        pngs = sorted(pngs)
        jsons = sorted(jsons)
        
        # Train/Valid 분할
        self.filenames, self.labelnames = self._split_dataset(
            pngs, jsons, is_train, n_splits, fold
        )
        
        print(f"{'Train' if is_train else 'Valid'} dataset: {len(self.filenames)} samples")
    
    def _collect_files(self, root: str, extension: str) -> list:
        """지정된 확장자의 파일들을 재귀적으로 수집"""
        files = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(extension):
                    rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
                    files.append(rel_path)
        return files
    
    def _split_dataset(
        self,
        pngs: list,
        jsons: list,
        is_train: bool,
        n_splits: int,
        fold: int,
    ) -> tuple:
        """GroupKFold를 사용하여 데이터셋 분할"""
        _filenames = np.array(pngs)
        _labelnames = np.array(jsons)
        
        # 폴더명을 그룹으로 사용 (동일 인물 데이터 분리 방지)
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0] * len(_filenames)  # dummy labels
        
        gkf = GroupKFold(n_splits=n_splits)
        
        filenames = []
        labelnames = []
        
        for i, (train_idx, valid_idx) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train:
                if i == fold:
                    continue
                filenames.extend(_filenames[valid_idx])
                labelnames.extend(_labelnames[valid_idx])
            else:
                if i == fold:
                    filenames = list(_filenames[valid_idx])
                    labelnames = list(_labelnames[valid_idx])
                    break
        
        return filenames, labelnames
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # 이미지 로드
        image_path = os.path.join(self.image_root, self.filenames[idx])
        image = cv2.imread(image_path)
        image = image / 255.0
        
        # 라벨 로드
        label_path = os.path.join(self.label_root, self.labelnames[idx])
        label_shape = tuple(image.shape[:2]) + (len(self.classes),)
        label = np.zeros(label_shape, dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        for ann in annotations:
            c = ann["label"]
            if c not in self.class2ind:
                continue
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label
        
        # Transform 적용
        if self.transforms is not None:
            inputs = {"image": image, "mask": label}
            result = self.transforms(**inputs)
            image = result["image"]
            label = result["mask"]
        
        # Channel first로 변환
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        return image, label


class XRayInferenceDataset(Dataset):
    """
    추론용 X-Ray Dataset
    
    Args:
        image_root (str): 이미지 폴더 경로
        transforms: albumentations 변환
    """
    
    def __init__(self, image_root: str, transforms=None):
        self.image_root = image_root
        self.transforms = transforms
        
        # 이미지 파일 목록 수집
        pngs = []
        for dirpath, _, filenames in os.walk(image_root):
            for fname in filenames:
                if fname.lower().endswith(".png"):
                    rel_path = os.path.relpath(os.path.join(dirpath, fname), image_root)
                    pngs.append(rel_path)
        
        self.filenames = sorted(pngs)
        print(f"Test dataset: {len(self.filenames)} samples")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image_path = os.path.join(self.image_root, image_name)
        
        image = cv2.imread(image_path)
        image = image / 255.0
        
        if self.transforms is not None:
            result = self.transforms(image=image)
            image = result["image"]
        
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        
        return image, image_name

