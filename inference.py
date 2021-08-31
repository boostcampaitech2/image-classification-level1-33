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
    
    preds2class = {
        "000":0,
        "001":1,
        "002":2,
        "010":3,
        "011":4,
        "012":5,
        "100":6,
        "101":7,
        "102":8,
        "110":9,
        "111":10,
        "112":11,
        "200":12,
        "201":13,
        "202":14,
        "210":15,
        "211":16,
        "212":17
    }
    
    all_predictions = []
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

            pred_total = "" 
            for pred in [int(pred_mask),int(pred_gender),int(pred_age)]:
                pred_total += str(pred)
            #print(pred_total)

            pred_class = preds2class[pred_total]
            all_predictions.append(pred_class)

    submission['ans'] = all_predictions
    submission.to_csv('./submission.csv', index=False)
    print('test inference is done!')    
    
if __name__=='__main__':
    with open('./args.json','r') as f:
        args = easydict.EasyDict(json.load(f))
        
    inference(args)