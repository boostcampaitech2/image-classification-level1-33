# 33조 Image Classification
33조 이미지 분류 대회 github repository 입니다.

## 사용법
```bash
python train.py --seed=43
```

## Configuration / Arguments 설명

`seed` 고정된 시드를 설정합니다

### args.json에 대한 설명입니다.

1. "project" : "Wandb-Project-Name",   
 wandb 프로젝트명
2. "seed" : 1004,   
 seed는 args를 바꿔도 안바뀝니다.
3. "epochs" : 10,   
4. "dataset" : "combined",   
 데이터셋 종류 : "original", "aaf", "combined", "combined_test" 중 하나 선택
5. "target" : "agegroup",   
 라벨 종류 : "mask", "gender", "agegroup" 중 하나 선택
6. "augmentation_original" : "BaseAugmentationForOriginal",   
 dataset.py에 정의된 Augmentation Class 이름. 
7. "augmentation_aaf": "BaseAugmentationForAAF",   
 기존 데이터셋과 추가 데이터셋의 사진들이 좀 다르기때문에 각각 Augmentation 적용
8. "batch_size": 128,   
9. "model": "VGG",   
 훈련에 쓸 모델 클래스이름. model.py에 정의해 놓은 클래스 중 하나
10. "model_mask": "VGG",   
 inference 시 사용할 mask 모델
11. "model_gender": "VGG",   
 inference 시 사용할 gender 모델
12. "model_age": "RESNET152",   
 inference 시 사용할 agegroup 모델
13. "model_mask_dir": "./results/mask/005_acc99.92%.ckpt",   
 inference 시 사용할 모델의 state_dict. train하면서 폴더와 파일이 자동으로 생성됨.
14. "model_gender_dir": "./results/gender/012_f197.44%.ckpt",   
15. "model_age_dir": "./results/age/013_f177.78%.ckpt",   
16. "optimizer": "Adam",   
17. "lr": 0.0001,   
18. "criterion": "cross_entropy"   

