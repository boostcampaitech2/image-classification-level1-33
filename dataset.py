from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import *
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import pandas as pd
import numpy as np

from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from fractions import Fraction as frac
from pandas_streaming.df import train_test_apart_stratify

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

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



class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")
    
    @classmethod
    def from_id(cls, value: str) -> int:
        if int(value) <= 7380:
            return cls.FEMALE
        elif int(value) > 7380:
            return cls.MALE
        else:
            raise ValueError(f"Gender value from id should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 57:
            return cls.MIDDLE
        else:
            return cls.OLD

class CustomDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    all_labels = []#-
    indexes = []#-
    groups = []#-

    def __init__(self, target, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.org_dir = '/opt/ml/input/crop_train_images'
        self.aaf_dir = '/opt/ml/input/crop_aaf_images'
        # self.org_dir = '/opt/ml/input/data/train/images'
        # self.aaf_dir = '/opt/ml/input/data2/images/aligned'
        
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
        
        self.target = target

        if target == 'mask':
            self.classes_num = 3 #['wear', 'incorrect', 'normal']
        elif target == 'gender':
            self.classes_num = 2 #['male', 'female']
        elif target == 'agegroup':
            self.classes_num = 3 #['young', 'middle', 'old']
        


    def setup(self):
        cnt = 0#-
        org_profiles = os.listdir(self.org_dir)
        for profile in org_profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.org_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.org_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))#-
                self.indexes.append(cnt)#-
                self.groups.append(id)#-
                cnt += 1#-
                
        aaf_profiles = os.listdir(self.aaf_dir)
        for file_name in aaf_profiles:
            if file_name.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue            

            img_path = os.path.join(self.aaf_dir, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
            mask_label = self._file_names["normal"]

            id, age = os.path.splitext(file_name)[0].split('A')
            gender_label = GenderLabels.from_id(id)
            age_label = AgeLabels.from_number(age)
            
            self.image_paths.append(img_path)
            self.mask_labels.append(mask_label)
            self.gender_labels.append(gender_label)
            self.age_labels.append(age_label)
            self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))#-
            self.indexes.append(cnt)#-
            self.groups.append(id+'a')#-
            cnt += 1#-
        

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        
        image_transform = self.transform(image)
        
        if self.target == 'mask' :
            return_classes = mask_label
        elif self.target == 'gender' :
            return_classes = gender_label
        elif self.target == 'agegroup' :
            return_classes = age_label
             
        return image_transform, return_classes

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'agegroup' :
            self.classes = self.age_labels
        
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)#-
        train_index = train["idxs"].tolist()#-
        valid_index = valid["idxs"].tolist()#-
        
        return [Subset(self, train_index), Subset(self, valid_index)]
    
    def split_dataset_Kfold(self, k) -> Tuple[Subset, Subset]:
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'agegroup' :
            self.classes = self.age_labels
            
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        fold_val_ratio = frac(1, int(k)).limit_denominator()
        tmp_df = df.copy()
        fold_trn = []
        fold_val = []
        
        for i in range(1, int(str(frac(fold_val_ratio).limit_denominator())[-1])):
            tmp_df, tmp_val = train_test_apart_stratify(tmp_df, group="groups", stratify="labels", test_size=fold_val_ratio)#-

            tmp_trn = df.drop(df.index[tmp_val.index])
            fold_trn.append(tmp_trn["idxs"].tolist())
            fold_val.append(tmp_val["idxs"].tolist())

            fold_val_ratio = float(frac(1, int(str(frac(fold_val_ratio).limit_denominator())[-1])-1))
            if fold_val_ratio == 1:
                tmp_val = tmp_df
                tmp_trn = df.drop(df.index[tmp_val.index])
                fold_trn.append(tmp_trn["idxs"].tolist())
                fold_val.append(tmp_val["idxs"].tolist())

        return [[Subset(self, train_index), Subset(self, valid_index)] for train_index, valid_index in zip(fold_trn, fold_val)]

class CustomDataset_Pseudo(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    all_labels = []#-
    indexes = []#-
    groups = []#-

    def __init__(self, target, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.org_dir = '/opt/ml/input/crop_train_images'
        self.aaf_dir = '/opt/ml/input/crop_aaf_images'
        # self.org_dir = '/opt/ml/input/data/train/images'
        # self.aaf_dir = '/opt/ml/input/data2/images/aligned'
        self.test_dir = '/opt/ml/input/crop_eval_images'
        self.test_data = pd.read_csv("csv/df_train_for_test_idx.csv")

        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
        
        self.target = target

        if target == 'mask':
            self.classes_num = 3 #['wear', 'incorrect', 'normal']
        elif target == 'gender':
            self.classes_num = 2 #['male', 'female']
        elif target == 'agegroup':
            self.classes_num = 3 #['young', 'middle', 'old']
        


    def setup(self):
        cnt = 0#-
        org_profiles = os.listdir(self.org_dir)
        for profile in org_profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.org_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.org_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))#-
                self.indexes.append(cnt)#-
                self.groups.append(id)#-
                cnt += 1#-

        for i, file_name in enumerate(self.test_data):          

            img_path = os.path.join(self.test_dir, file_name[i]['filename'])  # (resized_data, 000004_male_Asian_54, mask1.jpg)
            mask_label = file_name[i]['mask']

            id = file_name[i]['filename']
            gender_label = file_name[i]['gender']
            age_label = file_name[i]['agegroup']
            
            self.image_paths.append(img_path)
            self.mask_labels.append(mask_label)
            self.gender_labels.append(gender_label)
            self.age_labels.append(age_label)
            self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))#-
            self.indexes.append(cnt)#-
            self.groups.append(id+'t')#-
            cnt += 1#-
                
        aaf_profiles = os.listdir(self.aaf_dir)
        for file_name in aaf_profiles:
            if file_name.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue            

            img_path = os.path.join(self.aaf_dir, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
            mask_label = self._file_names["normal"]

            id, age = os.path.splitext(file_name)[0].split('A')
            gender_label = GenderLabels.from_id(id)
            age_label = AgeLabels.from_number(age)
            
            self.image_paths.append(img_path)
            self.mask_labels.append(mask_label)
            self.gender_labels.append(gender_label)
            self.age_labels.append(age_label)
            self.all_labels.append(self.encode_multi_class(mask_label, gender_label, age_label))#-
            self.indexes.append(cnt)#-
            self.groups.append(id+'a')#-
            cnt += 1#-
        

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
        
        image_transform = self.transform(image)
        
        if self.target == 'mask' :
            return_classes = mask_label
        elif self.target == 'gender' :
            return_classes = gender_label
        elif self.target == 'agegroup' :
            return_classes = age_label
             
        return image_transform, return_classes

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'agegroup' :
            self.classes = self.age_labels
        
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)#-
        train_index = train["idxs"].tolist()#-
        valid_index = valid["idxs"].tolist()#-
        
        return [Subset(self, train_index), Subset(self, valid_index)]
    
    def split_dataset_Kfold(self, k) -> Tuple[Subset, Subset]:
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'agegroup' :
            self.classes = self.age_labels
            
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        fold_val_ratio = frac(1, int(k)).limit_denominator()
        tmp_df = df.copy()
        fold_trn = []
        fold_val = []
        
        for i in range(1, int(str(frac(fold_val_ratio).limit_denominator())[-1])):
            tmp_df, tmp_val = train_test_apart_stratify(tmp_df, group="groups", stratify="labels", test_size=fold_val_ratio)#-

            tmp_trn = df.drop(df.index[tmp_val.index])
            fold_trn.append(tmp_trn["idxs"].tolist())
            fold_val.append(tmp_val["idxs"].tolist())

            fold_val_ratio = float(frac(1, int(str(frac(fold_val_ratio).limit_denominator())[-1])-1))
            if fold_val_ratio == 1:
                tmp_val = tmp_df
                tmp_trn = df.drop(df.index[tmp_val.index])
                fold_trn.append(tmp_trn["idxs"].tolist())
                fold_val.append(tmp_val["idxs"].tolist())

        return [[Subset(self, train_index), Subset(self, valid_index)] for train_index, valid_index in zip(fold_trn, fold_val)]
