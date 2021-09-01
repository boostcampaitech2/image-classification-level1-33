# 33조 Image Classification
33조 이미지 분류 대회 github repository 입니다.

## 사용법
```bash
python train.py --seed=43
```

## Configuration / Arguments 설명

`seed` 고정된 시드를 설정합니다

args.json에 대한 설명입니다.

"project": "Wandb-Project-Name", <- wandb 프로젝트명
"seed": 42,                     
"epochs": 10,
"dataset": "combined",           <- 데이터셋 종류 : "original", "aaf", "combined" 중 하나 선택
"target": "agegroup",            <- 라벨 종류 : "mask", "gender", "agegroup" 중 하나 선택
"augmentation_original": "BaseAugmentationForOriginal",      <- dataset.py에 정의된 Augmentation Class 이름. 
"augmentation_aaf": "BaseAugmentationForAAF",                <- 기존 데이터셋과 추가 데이터셋의 사진들이 좀 다르기때문에 각각 Augmentation 적용
"batch_size": 128,
"model": "VGG",                  <- 훈련에 쓸 모델 클래스이름. model.py에 정의해 놓은 클래스 중 하나
"model_mask": "VGG",             <- inference 시 사용할 mask 모델
"model_gender": "VGG",           <- inference 시 사용할 gender 모델
"model_age": "RESNET152",        <- inference 시 사용할 agegroup 모델
"model_mask_dir": "./results/mask/005_acc99.92%.ckpt",     <- inference 시 사용할 모델의 state_dict. train하면서 폴더와 파일이 자동으로 생성됨.
"model_gender_dir": "./results/gender/012_f197.44%.ckpt",
"model_age_dir": "./results/age/013_f177.78%.ckpt",
"optimizer": "Adam",
"lr": 0.0001,
"criterion": "cross_entropy"

