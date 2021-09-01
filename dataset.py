from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

import os
import pandas as pd

ORIGINAL_DATA_DIR = '/opt/ml/input/data/train/images'
AAF_DATA_DIR = '/opt/ml/input/data2/images'
TEST_DATA_DIR = '/opt/ml/input/data/eval'

path = {
    'original': ORIGINAL_DATA_DIR,
    'aaf': AAF_DATA_DIR,
    'test' : TEST_DATA_DIR
}

class MaskDataset(Dataset):
    def __init__(self, path, dataset,target,train,transform):
        '''
        **인자 설명**
        path: dict 객체. 키(key) 'original'과 'aaf'에 대해 -> 실제 이미지가 있는 디렉토리의 직전 디렉토리  (ex. dir1/dir2/image.jpg 에서 dir1까지)
        dataset: 'original', 'aaf', 'combined' 중 하나 선택.
        target: 'mask', 'gender', 'agegroup' 중 하나 선택
        train: bool값. train set인지 validation set인지.
        '''
        
        self.path = path
        self.train = train
        self.transform = transform
        self.target = target
        
        if target=='mask':
            self.classes = ['wear','incorrect','normal']
        elif target =='gender':
            self.classes = ['male','female']
        else:
            self.classes = ['young','middle','old']
            
        
        
        if self.train:
            self.data = pd.read_csv(f"csv/df_train_{dataset}.csv")
        else:
            self.data = pd.read_csv(f"csv/df_valid_{dataset}.csv")
            
        self.count = [(self.data[target]==cls).sum() for cls in self.classes] # 클래스별 데이터 수 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        item = self.data.iloc[idx]
        dataset = item.dataset # 'original' or 'aaf'
        
        image = Image.open(os.path.join(path[dataset],item.folder,item.filename))
        if self.target=='mask':   label = self.classes.index(item['mask'])
        if self.target=='gender': label = self.classes.index(item['gender'])
        if self.target=='agegroup':    label = self.classes.index(item['agegroup'])

                           
        if self.transform:
            image = self.transform[dataset](image)
                           
        return image, label
    
    def set_transform(self, transform):
        self.transform = transform
        
class BaseAugmentationForOriginal:
    def __init__(self):
        self.transform = transforms.Compose([
            CenterCrop(350),
            Resize((224,224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
    def __call__(self, image):
        return self.transform(image)
    
class BaseAugmentationForAAF:
    def __init__(self):
        self.transform = transforms.Compose([
            Resize((224,224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
    def __call__(self, image):
        return self.transform(image)

class BaseAugmentationForTEST:
    def __init__(self):
        self.transform = transforms.Compose([
            Resize((224,224), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
    def __call__(self, image):
        return self.transform(image)
    
def load_dataset(dataset,target,train):
    '''
    dataset : 'original', 'aaf', 'combined' 중 선택
    target  : 'mask', 'gender', 'agegroup' 중 선택
    train   : True면 Train 셋, False면 Validation 셋
    '''
    transform_original = BaseAugmentationForOriginal()
    transform_aaf = BaseAugmentationForAAF()
    transform_test = BaseAugmentationForTEST()
    transform = {
        'original': transform_original,
        'aaf': transform_aaf,
        'test' : transform_test
    }
    
    print("loading dataset...")
    return MaskDataset(path, dataset,target,train,transform)