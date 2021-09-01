from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import pandas as pd
import numpy as np

ORIGINAL_DATA_DIR = '/opt/ml/input/data/train/images'
AAF_DATA_DIR = '/opt/ml/input/data2/images'
TEST_DATA_DIR = '/opt/ml/input/data/eval'

path = {
    'original': ORIGINAL_DATA_DIR,
    'aaf': AAF_DATA_DIR,
    'test': TEST_DATA_DIR  # for psudo labeling
}


class MaskDataset(Dataset):
    def __init__(self, path, dataset, target, train, transform):
        '''
        **인자 설명**
        path: dict 객체. 키(key) 'original'과 'aaf'에 대해 -> 실제 이미지가 있는 디렉토리의 직전 디렉토리  (ex. dir1/dir2/image.jpg 에서 dir1까지)
        dataset: 'original', 'aaf', 'combined' 중 하나 선택.
        target: 'mask', 'gender', 'agegroup' 중 하나 선택
        train: bool값. train set인지 validation set인지.
        '''

        self.path = path
        self.train = train
        self.transform = transform  # 여기서 trn-o,a tst-o, a 가져오고
        self.target = target

        if target == 'mask':
            self.classes = ['wear', 'incorrect', 'normal']
        elif target == 'gender':
            self.classes = ['male', 'female']
        else:
            self.classes = ['young', 'middle', 'old']

        if self.train:
            self.data = pd.read_csv(f"csv/df_train_{dataset}.csv")
        else:
            self.data = pd.read_csv(f"csv/df_valid_{dataset}.csv")

        self.count = [(self.data[target] == cls).sum()
                      for cls in self.classes]  # 클래스별 데이터 수

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]  # row
        dataset = item.dataset  # 'original' or 'aaf'

        image = Image.open(os.path.join(
            path[dataset], item.folder, item.filename))
        if self.target == 'mask':
            label = self.classes.index(item['mask'])
        if self.target == 'gender':
            label = self.classes.index(item['gender'])
        if self.target == 'agegroup':
            label = self.classes.index(item['agegroup'])
        image = np.array(image)
        if self.transform:
            if self.train:
                # albumentation에서만 동작하는 코드입니다.
                image = self.transform[f'{dataset}_trn'].transform(image=image)
            else:
                # albumentation에서만 동작하는 코드입니다.
                image = self.transform[f'{dataset}_tst'].transform(image=image)
             # 여기서 if 문으로 잘 설정해주자 original_trn, aaf_trn
            image = image['image']  # albumentation 특징
        return image, label

    def set_transform(self, transform):
        self.transform = transform


class BaseAugmentationForOriginal:
    def __init__(self):
        self.transform = transforms.Compose([
            CenterCrop(350),
            Resize((224, 224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])

    def __call__(self, image):
        return self.transform(image)


class BaseAugmentationForAAF:
    def __init__(self):
        self.transform = transforms.Compose([
            Resize((224, 224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])

    def __call__(self, image):
        return self.transform(image)


class AlbumentationForOriginalTrn():
    def __init__(self):
        self.transform = A.Compose([
            A.CenterCrop(350, 350),
            A.Resize(224, 224),
            A.OneOf([
                A.GaussNoise(var_limit=(1000, 1600), p=1.0),
                A.GlassBlur(p=1.0),
                A.Cutout(num_holes=16, max_h_size=10,
                         max_w_size=10, fill_value=0, p=1.0)
            ]),
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform()


class AlbumentationForAAFTrn():
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.OneOf([
                A.GaussNoise(var_limit=(1000, 1600), p=1.0),
                A.GlassBlur(p=1.0),
                A.Cutout(num_holes=16, max_h_size=10,
                         max_w_size=10, fill_value=0, p=1.0)
            ]),
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform()


class AlbumentationForOriginalTst():
    def __init__(self):
        self.transform = A.Compose([
            A.CenterCrop(350, 350),
            A.Resize(224, 224),
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform()


class AlbumentationForAAFTst():
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
            ToTensorV2()
        ])

    def __call__(self, image):
        return self.transform()


class BaseAugmentationForTEST:
    def __init__(self):
        self.transform = transforms.Compose([
            Resize((224, 224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])

    def __call__(self, image):
        return self.transform(image)


def load_dataset(dataset, target, train):
    '''
    dataset : 'original', 'aaf', 'combined' 중 선택
    target  : 'mask', 'gender', 'agegroup' 중 선택
    train   : True면 Train 셋, False면 Validation 셋
    '''

    transform_original_trn = AlbumentationForOriginalTrn()
    transform_original_tst = AlbumentationForOriginalTst()
    transform_aaf_trn = AlbumentationForAAFTrn()
    transform_aaf_tst = AlbumentationForAAFTst()
    transform_test = AlbumentationForOriginalTrn()

    transform = {
        'original_trn': transform_original_trn,
        'original_tst': transform_original_tst,
        'aaf_trn': transform_aaf_trn,
        'aaf_tst': transform_aaf_tst,
        'test_trn': transform_test
    }

#     transform_original = BaseAugmentationForOriginal()
#     transform_aaf = BaseAugmentationForAAF()
#     transform_test = BaseAugmentationForTEST()
#     transform = {
#         'original': transform_original,
#         'aaf': transform_aaf}

    print("loading dataset...")
    return MaskDataset(path, dataset, target, train, transform)
