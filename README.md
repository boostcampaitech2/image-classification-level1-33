# 33조 Image Classification
33조 이미지 분류 대회 github repository 입니다.

## 사용법
```bash
python train.py
or
python train_kfold.py
```

### args.json에 대한 설명입니다.
학습시 사용하는 인자에 대한 설명입니다.

```
project : "Wandb-Project-Name"
 wandb 프로젝트명

seed : 1004   
 seed는 args를 바꿔도 안바뀝니다.
 
epochs : 10,   

dataset : "combined"   
 데이터셋 종류 : "original", "aaf", "combined", "combined_test" 중 하나 선택
 
target : "agegroup"   
 라벨 종류 : "mask", "gender", "agegroup" 중 하나 선택
 
augmentation_original : "BaseAugmentationForOriginal"  
 dataset.py에 정의된 Augmentation Class 이름. 
 
augmentation_aaf : "BaseAugmentationForAAF"  
 기존 데이터셋과 추가 데이터셋의 사진들이 좀 다르기때문에 각각 Augmentation 적용
 
batch_size : 64   

model : "ResNet120"   
 훈련에 쓸 모델 클래스이름. model.py에 정의해 놓은 클래스 중 하나
 
model_mask : "ResNet120"
 inference 시 사용할 mask 모델
 
model_gender : "ResNet120"   
 inference 시 사용할 gender 모델
 
model_age : "RESNET152"   
 inference 시 사용할 agegroup 모델
 
model_mask_dir : "./results/mask/005_acc99.92%.ckpt"  
 inference 시 사용할 모델의 state_dict. train하면서 폴더와 파일이 자동으로 생성됨.
 
model_gender_dir : "./results/gender/012_f197.44%.ckpt" 

model_age_dir : "./results/age/013_f177.78%.ckpt" 

optimizer : "Adam"   

lr : 0.0001

lr_scheduler : LambdaLR(lr_lambda=lambda epoch: 0.95**epoch)

criterion : "cross_entropy"

kfold_num : 5
```


<br>

## Structure

```python
├── README.md
├── args.json
├── dataset.py
├── dataset_final_edit.py
├── ensemble.py
├── inference.py
├── loss.py
├── model.py
├── requirements.txt
├── train.py
├── train_kfold.py
```
**args.json** : train시 필요한 arguments입니다.<br><br>
**dataset.py** : train, validation 데이터를 csv 파일을 바탕으로 로드하는 dataset클래스와 transform을 정의합니다.<br><br>
**dataset_final_edit.py** : 전체 데이터를 로드한 후 train, validation을 나누는 dataset클래스와 transform을 정의합니다. <br><br>
**ensemble.py** : target(agegroup, mask, gender) folds에 입력된 모델들의 soft voting을 진행합니다. <br><br>
**inference.py** : Inference, 최종 제출 submisson.csv를 만듭니다.<br><br>
**loss.py** : Label smoothing loss를 정의합니다.<br><br>
**model.py** : ResNet152, VGG_bn 모델을 정의합니다. <br><br>
**train.py** : train 과정을 진행합니다.<br><br>
**train_kfold.py** : kfold가 적용된 train을 진행합니다. <br><br>
<br>

## Contributors
BoostCamp AI TECH Level1-Ustage 33조팀원들입니다. 

<table>
  <tr>
    <td align="center"><a href="https://github.com/jiwoo0212"><img src="https://user-images.githubusercontent.com/67720742/125877217-f8d4d731-e5a9-41f6-8820-5223a4d6b0c6.jpg" width="150" height="150"><br /><sub><b>강지우</b></sub></td>
    <td align="center"><a href="https://blog.naver.com/7tkfkd"><img src="https://ifh.cc/g/puHQTP.jpg" width="150" height="150"><br /><sub><b>김성민</b></sub></td>
     <td align="center"><a href="https://github.com/jiwoo0212"><img src="https://user-images.githubusercontent.com/67720742/125877217-f8d4d731-e5a9-41f6-8820-5223a4d6b0c6.jpg" width="150" height="150"><br /><sub><b>남세현</b></sub></td>
  </tr>
</table>

<table>
  <tr>
       <td align="center"><a href="https://github.com/uyeongjae"><img src="https://avatars.githubusercontent.com/u/53523319?v=4" width="150" height="150"><br /><sub><b>유영재</b></sub></td>
    <td align="center"><a href="https://github.com/pseeej"><img src="https://user-images.githubusercontent.com/49185035/132097668-01e941dc-a1db-47f6-a209-f3385249ecfb.png" width="150" height="150"><br /><sub><b>박세진</b></sub></td>
     <td align="center"><a href="https://github.com/jiwoo0212"><img src="https://user-images.githubusercontent.com/67720742/125877217-f8d4d731-e5a9-41f6-8820-5223a4d6b0c6.jpg" width="150" height="150"><br /><sub><b>정세종</b></sub></td>

