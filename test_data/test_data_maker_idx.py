import os
import json
import easydict

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, tqdm_notebook

from importlib import import_module
from torchvision import transforms, models
from torchvision.transforms import Resize, ToTensor, Normalize

device = torch.device('cuda')

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

def inference(args):
    # 테스트 데이터셋 폴더 경로
    test_dir = '/opt/ml/input/data/eval'

    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    
    transform = getattr(import_module('dataset'), args.augmentation_original)()
    
    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset,shuffle=False)
    print("데이터셋 로드 완료")
    
    # model 불러오기
    model_mask = getattr(import_module('model'), args.model_mask)(num_class=3)
    model_gender = getattr(import_module('model'), args.model_gender)(num_class=2)
    model_age = getattr(import_module('model'), args.model_age)(num_class=3)
    print(f"model_mask: {args.model_mask}")
    print(f"model_gender: {args.model_gender}")
    print(f"model_age: {args.model_age}")
    # state_dict 불러오기
    model_mask.load_state_dict(torch.load(args.model_mask_dir))
    model_gender.load_state_dict(torch.load(args.model_gender_dir))
    model_age.load_state_dict(torch.load(args.model_age_dir))
    print("저장된 모델 불러오기 완료")
    model_mask = model_mask.to(device)
    model_gender = model_gender.to(device)
    model_age = model_age.to(device)

    model_mask.eval()
    model_gender.eval()
    model_age.eval()
    
    gender_predictions = []
    age_predictions = []
    mask_predictions = []

    for image in tqdm(loader):
        #print(image.shape)
        with torch.no_grad():
            image = image.to(device)
            out_mask = model_mask(image)
            out_gender = model_gender(image)
            out_age = model_age(image)

            pred_mask = out_mask.argmax(dim=-1).cpu().numpy()
            pred_gender = out_gender.argmax(dim=-1).cpu().numpy()
            pred_age = out_age.argmax(dim=-1).cpu().numpy()

            gender_predictions.append(int(pred_gender[0]))
            mask_predictions.append(int(pred_mask[0]))
            age_predictions.append(int(pred_age[0]))

    df = pd.DataFrame({'dataset' : ['test' for _ in range(len(image_paths))],
                        'agegroup' : age_predictions,
                        'gender' : gender_predictions,
                        'mask' : mask_predictions,
                        'forder' : ['images' for _ in range(len(image_paths))],
                        'filename': [i for i in submission.ImageID]
                        })

    df.to_csv('./for_test_idx.csv', index=False)
    print('test data making is done!') 
    

    
if __name__=='__main__':
    with open('./args.json','r') as f:
        args = easydict.EasyDict(json.load(f))
        
    inference(args)