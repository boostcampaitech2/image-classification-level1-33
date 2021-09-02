import os
import json
from albumentations.pytorch.transforms import img_to_tensor
import easydict

import pandas as pd
from PIL import Image
import numpy as np

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
            image = np.array(image)
            image = self.transform(image=image)
            image = image['image']
        return image

    def __len__(self):
        return len(self.img_paths)

def get_ensemble_models(target):
    models_name_list = []
    model_folder_path = './results/'
    model_save_path = './results'
    model_folder_names = [m_fold_name for m_fold_name in os.listdir(model_save_path) if target in m_fold_name]

    for i, model_folder in enumerate(model_folder_names):
        model_path = os.path.join(model_save_path, model_folder)
        max_f1 = max([os.path.splitext(model_name)[0][-6:-1] for model_name in os.listdir(model_path)])
        max_f1_model_name = os.path.join(model_path, [model_name for model_name in os.listdir(model_path) if max_f1 in model_name][0])

        models_name_list.append(max_f1_model_name)

    return models_name_list

def inference(args):
    # 테스트 데이터셋 폴더 경로
    test_dir = '/opt/ml/input/crop_eval_images'
    
    # k_fold 별 max_f1 model 경로
    k_model_list = get_ensemble_models(args.target)

    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    image_dir = test_dir

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id)
                   for img_id in submission.ImageID]

    transform = getattr(import_module('dataset'),
                        args.augmentation_test)()
    transform = transform.transform

    dataset = TestDataset(image_paths, transform)
    test_loader = DataLoader(dataset, shuffle=False)
    print("데이터셋 로드 완료")

    # ensemble model load
    ensemble_pred = []
    for model_name in k_model_list:
        model_mask = getattr(import_module('model'), args.model_mask)(num_class=3)
        model_mask.load_state_dict(torch.load(model_name))
        model_mask = model_mask.to(device)
        model_mask.eval()

        for test in tqdm(test_loader):
            # print(image.shape)
            with torch.no_grad():
                test = test.to(device)
                out_mask = model_mask(test)
                
                soft_mask = F.softmax(out_mask, dim=1).cpu().numpy()
                ensemble_pred.append(soft_mask)

    pred = np.array(ensemble_pred)
    pred = pred.reshape(5, -1, 3)
    soft_voting_pred = pred.sum(axis=0)
    voting_result = soft_voting_pred.argmax(axis=-1)

    submission['ans'] = voting_result
    submission.to_csv('./agegroup_result.csv', index=False)
    print('test inference is done!')


if __name__ == '__main__':
    with open('./args.json', 'r') as f:
        args = easydict.EasyDict(json.load(f))

    inference(args)