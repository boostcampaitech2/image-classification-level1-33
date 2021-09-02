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


def inference(args):
    # 테스트 데이터셋 폴더 경로
    test_dir_info = '/opt/ml/input/data/eval'

    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir_info, 'info.csv'))
    image_dir = '/opt/ml/input/crop_eval_images'

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id)
                   for img_id in submission.ImageID]

    transform = getattr(import_module('dataset'),
                        args.augmentation_original)()
    transform = transform.transform

    dataset = TestDataset(image_paths, transform)
    loader = DataLoader(dataset, shuffle=False)
    print("데이터셋 로드 완료")

    # model 불러오기
    model_age = getattr(import_module('model'), args.model_age)(num_class=3)
    print(f"model_age: {args.model_age}")
    # state_dict 불러오기
    model_age.load_state_dict(torch.load(args.model_age_dir))
    print("저장된 모델 불러오기 완료")
    model_age = model_age.to(device)

    model_age.eval()

    age_predictions = []
    for image in tqdm(loader):
        # print(image.shape)
        with torch.no_grad():
            image = image.to(device)

            out_age = model_age(image)

            pred_age = out_age.argmax(dim=-1).cpu().numpy()

            age_predictions.append(int(pred_age))

    submission['agegroup'] = age_predictions
    submission.to_csv('./submission_age.csv', index=False)
    print('test inference is done!')


if __name__ == '__main__':
    with open('./args.json', 'r') as f:
        args = easydict.EasyDict(json.load(f))

    inference(args)
