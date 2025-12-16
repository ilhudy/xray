# 🦴 X-Ray Bone Segmentation

손 X-Ray 이미지에서 뼈를 29개 클래스로 세그멘테이션하는 딥러닝 프로젝트입니다.

## 📋 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [환경 설정](#환경-설정)
3. [빠른 시작](#빠른-시작)
4. [실험 진행 방법](#실험-진행-방법)
5. [설정 파일 가이드](#설정-파일-가이드)
6. [WandB 사용법](#wandb-사용법)
7. [협업 규칙](#협업-규칙)
8. [트러블슈팅](#트러블슈팅)
9. [실험 체크리스트](#실험-체크리스트)

---

## 📁 프로젝트 구조

```
Chest-Xray-Segmentation/
├── .gitignore          # Git 무시 파일 목록
├── README.md           # 이 파일
├── requirements.txt    # 필요 라이브러리 목록
├── main.py             # 학습/추론 메인 코드
├── configs/            # 실험 설정 파일 (YAML)
│   ├── base_config.yaml         # 기본 설정 템플릿
│   ├── exp01_fcn_resnet50.yaml  # 실험 1: FCN 베이스라인
│   ├── exp02_loss_experiment.yaml    # 실험 2: Loss 실험
│   └── exp03_augmentation.yaml  # 실험 3: Augmentation 실험
├── src/                # 핵심 코드 모듈
│   ├── __init__.py
│   ├── dataset.py      # 데이터셋 클래스
│   ├── models.py       # 모델 정의
│   ├── losses.py       # Loss 함수
│   └── utils.py        # 유틸리티 함수
├── notebooks/          # (생성 필요) EDA, 개인 실험용 노트북
├── data/               # (Git 무시) 데이터셋 심볼릭 링크 또는 복사
└── saved_models/       # (Git 무시) 학습된 모델 저장
```

---

## ⚙️ 환경 설정

### 1. 저장소 클론

```bash
git clone [repository-url]
cd Chest-Xray-Segmentation
```

### 2. 가상환경 생성 (권장)

```bash
# conda 사용시
conda create -n xray python=3.10
conda activate xray

# 또는 venv 사용시
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. 라이브러리 설치

```bash
pip install -r requirements.txt

# PyTorch GPU 버전 설치 (CUDA 11.8 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. 데이터 경로 설정

데이터는 Git에 올리지 않으므로 **심볼릭 링크** 또는 **config 파일 수정**으로 연결합니다.

```bash
# 방법 1: 심볼릭 링크 생성
ln -s /path/to/Segmentation/train data/train
ln -s /path/to/Segmentation/test data/test

# 방법 2: config 파일에서 직접 경로 수정
# configs/base_config.yaml의 data 섹션 수정
```

---

## 🚀 빠른 시작

### 학습 실행

```bash
# 기본 실험 (FCN ResNet50)
python main.py --config configs/exp01_fcn_resnet50.yaml --mode train

# Loss 실험 (BCE + Dice)
python main.py --config configs/exp02_loss_experiment.yaml --mode train

# Augmentation 실험
python main.py --config configs/exp03_augmentation.yaml --mode train
```

### 추론 실행

```bash
# 학습된 모델로 추론
python main.py --config configs/exp01_fcn_resnet50.yaml --mode inference
```

### 학습 + 추론 한번에

```bash
python main.py --config configs/exp01_fcn_resnet50.yaml --mode all
```

---

## 🧪 실험 진행 방법

### Step 1: 새 실험 설정 파일 생성

```bash
# base_config.yaml을 복사하여 새 실험 파일 생성
cp configs/base_config.yaml configs/exp04_your_experiment.yaml
```

### Step 2: 설정 파일 수정

```yaml
# configs/exp04_your_experiment.yaml

experiment:
  name: "exp04_deeplabv3"           # ✏️ 실험 이름 수정
  description: "DeepLabV3 실험"      # ✏️ 설명 추가
  author: "your_name"               # ✏️ 본인 이름

model:
  name: "deeplabv3_resnet101"       # ✏️ 모델 변경
  pretrained: true

loss:
  name: "bce_dice"                  # ✏️ Loss 변경
  bce_weight: 0.5
  dice_weight: 0.5

save:
  dir: "saved_models/exp04_deeplabv3"  # ✏️ 저장 폴더 수정
```

### Step 3: 학습 실행

```bash
python main.py --config configs/exp04_your_experiment.yaml --mode train
```

### Step 4: 결과 기록

실험 결과를 팀 노션이나 스프레드시트에 기록합니다.

---

## 📝 설정 파일 가이드

### 🏗️ 사용 가능한 모델

| 모델명 | 설명 | 메모리 사용량 |
|--------|------|--------------|
| `fcn_resnet50` | FCN with ResNet50 | 낮음 |
| `fcn_resnet101` | FCN with ResNet101 | 중간 |
| `deeplabv3_resnet50` | DeepLabV3 with ResNet50 | 중간 |
| `deeplabv3_resnet101` | DeepLabV3 with ResNet101 | 높음 |
| `deeplabv3_mobilenet` | DeepLabV3 with MobileNet | 낮음 |

```yaml
model:
  name: "deeplabv3_resnet101"  # 원하는 모델 선택
  pretrained: true
```

### 📉 사용 가능한 Loss 함수

| Loss명 | 설명 | 사용 상황 |
|--------|------|----------|
| `bce` | Binary Cross Entropy | 기본 베이스라인 |
| `dice` | Dice Loss | 클래스 불균형 |
| `bce_dice` | BCE + Dice 조합 | 안정적인 학습 |
| `focal` | Focal Loss | 심한 클래스 불균형 |
| `tversky` | Tversky Loss | FP/FN 가중치 조절 |

```yaml
# BCE만 사용
loss:
  name: "bce"

# BCE + Dice 조합
loss:
  name: "bce_dice"
  bce_weight: 0.5    # BCE 가중치
  dice_weight: 0.5   # Dice 가중치

# Focal Loss
loss:
  name: "focal"
  alpha: 0.25        # 양성 클래스 가중치
  gamma: 2.0         # focusing parameter

# Tversky Loss
loss:
  name: "tversky"
  alpha: 0.3         # FP 가중치
  beta: 0.7          # FN 가중치
```

### 🔄 Augmentation 옵션

```yaml
augmentation:
  train:
    resize: 512                # 이미지 크기
    horizontal_flip: false     # 좌우 반전 (X-ray는 주의)
    vertical_flip: false       # 상하 반전
    rotate: true               # 회전
    rotate_limit: 15           # 회전 각도 범위 (±15도)
    brightness_contrast: true  # 밝기/대비 조절
    elastic_transform: true    # Elastic 변형
    grid_distortion: true      # Grid 왜곡
  valid:
    resize: 512
```

### 🎓 학습 파라미터

```yaml
training:
  epochs: 50              # 총 에폭 수
  batch_size: 8           # 배치 크기 (GPU 메모리에 따라 조절)
  learning_rate: 0.0001   # 학습률
  weight_decay: 0.000001  # 가중치 감쇠
  optimizer: "adam"       # adam, adamw, sgd
  scheduler: "cosine"     # null, cosine, step
  val_every: 5            # 검증 주기
  num_workers: 4          # DataLoader 워커 수
  n_splits: 5             # K-Fold 수
  fold: 0                 # 사용할 fold (0~4)
```

---

## 📊 WandB 사용법

### 1. WandB 초기 설정

```bash
# wandb 로그인 (처음 한 번만)
wandb login

# API 키 입력 (https://wandb.ai/authorize 에서 확인)
```

### 2. Config 파일 설정

```yaml
# 📊 WandB 설정
wandb:
  enabled: true                       # false로 끄면 로깅 안함
  entity: "let_cv_03"                 # 팀 entity 이름 (변경 금지!)
  project: "segmentation"             # 프로젝트 이름 (변경 금지!)
  name: "gh_fcn_resnet50_epoch50"     # ⚠️ 실험 이름 (필수 수정!)
  tags: ["baseline", "fcn"]           # 태그 (선택)
  notes: "FCN 베이스라인 실험"          # 메모 (선택)
```

### 3. 실험 이름 네이밍 규칙

**형식: `이니셜_모델_추가정보`**

| 예시 | 설명 |
|------|------|
| `gh_fcn_resnet50_baseline` | 가현, FCN ResNet50 베이스라인 |
| `jh_deeplabv3_bce_dice` | 지훈, DeepLabV3 BCE+Dice Loss |
| `sm_unet_augmentation` | 수민, UNet Augmentation 실험 |
| `yj_fcn_cosine_lr` | 영준, FCN Cosine LR 스케줄러 |

### 4. WandB 대시보드 확인

실험 실행 후 아래 주소에서 결과 확인:
- **프로젝트 URL**: https://wandb.ai/let_cv_03/segmentation

### 5. 로깅되는 항목

| 항목 | 설명 |
|------|------|
| `train/step_loss` | 매 step loss |
| `train/epoch_loss` | 에폭 평균 loss |
| `train/learning_rate` | 현재 learning rate |
| `val/loss` | 검증 loss |
| `val/dice` | 검증 평균 Dice |
| `val_dice/{class}` | 클래스별 Dice |
| `best_dice` | 최고 Dice (summary) |
| `best_epoch` | 최고 성능 에폭 (summary) |

### 6. WandB 끄기

로컬 테스트시 wandb를 끄려면:

```yaml
wandb:
  enabled: false    # false로 변경
```

### 7. 자주 사용하는 WandB 기능

```bash
# 오프라인 모드로 실행 (인터넷 없을 때)
WANDB_MODE=offline python main.py --config configs/exp01.yaml --mode train

# 실행 후 나중에 sync
wandb sync ./wandb/offline-run-*
```

---

## 🤝 협업 규칙

### 1. Git 브랜치 전략

```bash
# 새로운 실험은 본인 브랜치에서 진행
git checkout -b exp/홍길동/loss_experiment

# 작업 완료 후 main에 merge
git checkout main
git merge exp/홍길동/loss_experiment
```

### 2. 커밋 메시지 규칙

```bash
# 형식: [타입] 내용

# 예시
git commit -m "[exp] exp04 Loss 실험 추가"
git commit -m "[fix] dataset 경로 버그 수정"
git commit -m "[feat] Cosine scheduler 추가"
git commit -m "[docs] README 사용법 업데이트"
```

### 3. 실험 설정 파일 네이밍

```
exp[번호]_[실험내용].yaml

# 예시
exp01_fcn_resnet50.yaml      # 모델 실험
exp02_loss_bce_dice.yaml     # Loss 실험
exp03_augmentation.yaml       # Augmentation 실험
exp04_scheduler_cosine.yaml   # 스케줄러 실험
```

### 4. 결과 공유

실험 후 아래 정보를 팀과 공유합니다:

| 항목 | 내용 |
|------|------|
| 실험 번호 | exp04 |
| 실험자 | 홍길동 |
| 변경 사항 | DeepLabV3 + Dice Loss |
| Best Dice | 0.9523 |
| 학습 시간 | 2시간 |
| 특이사항 | epoch 30에서 수렴 |

---

## 🔧 트러블슈팅

### CUDA Out of Memory

```yaml
# 배치 사이즈 줄이기
training:
  batch_size: 4  # 8 → 4

# 또는 이미지 크기 줄이기
image:
  size: 256  # 512 → 256
```

### 학습이 너무 느림

```yaml
# num_workers 늘리기
training:
  num_workers: 8  # 4 → 8

# 또는 가벼운 모델 사용
model:
  name: "deeplabv3_mobilenet"
```

### Dice Score가 오르지 않음

```yaml
# 1. Loss 변경
loss:
  name: "bce_dice"

# 2. Learning rate 조절
training:
  learning_rate: 0.00001  # 줄이기

# 3. Augmentation 추가
augmentation:
  train:
    rotate: true
    brightness_contrast: true
```

### 데이터 경로 에러

```bash
# config 파일의 경로 확인
data:
  train_image_root: "../Segmentation/train/DCM"
  # 실제 경로가 맞는지 확인!

# 상대 경로 대신 절대 경로 사용
data:
  train_image_root: "/home/user/data/Segmentation/train/DCM"
```

---

## ✅ 실험 체크리스트

새로운 실험 전 확인사항:

- [ ] 새 config 파일 생성했는가?
- [ ] 실험 이름(`experiment.name`)을 고유하게 설정했는가?
- [ ] 저장 폴더(`save.dir`)를 실험별로 다르게 설정했는가?
- [ ] 출력 CSV(`inference.output_csv`)를 실험별로 다르게 설정했는가?
- [ ] 실험자 이름(`experiment.author`)을 입력했는가?
- [ ] 데이터 경로가 올바른가?
- [ ] **WandB 설정**: `wandb.name`을 본인 이니셜 + 실험 정보로 설정했는가?
- [ ] **WandB 설정**: `wandb.tags`에 실험 키워드를 추가했는가?

---

## 📊 29개 클래스 목록

| 번호 | 클래스명 | 설명 |
|------|----------|------|
| 0-18 | finger-1 ~ finger-19 | 손가락 뼈 |
| 19 | Trapezium | 대능형골 |
| 20 | Trapezoid | 소능형골 |
| 21 | Capitate | 유두골 |
| 22 | Hamate | 유구골 |
| 23 | Scaphoid | 주상골 |
| 24 | Lunate | 월상골 |
| 25 | Triquetrum | 삼각골 |
| 26 | Pisiform | 두상골 |
| 27 | Radius | 요골 |
| 28 | Ulna | 척골 |

---

## 👥 팀원

| 이름 | 역할 | 담당 실험 |
|------|------|----------|
| - | - | - |
| - | - | - |
| - | - | - |
| - | - | - |
| - | - | - |

---

## 📚 참고 자료

- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [Albumentations 문서](https://albumentations.ai/docs/)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)

---

**Happy Experimenting! 🚀**

