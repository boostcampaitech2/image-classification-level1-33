import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from fractions import Fraction as frac

import numpy as np
import pandas as pd
from pandas_streaming.df import train_test_apart_stratify
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


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


class MaskBaseDataset(Dataset):
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

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
        

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
        return image_transform, multi_class_label

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
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set
    

class CustomDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "wear" : MaskLabels.MASK,
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

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        cnt = 0#-
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.all_labels.append(self.encode_multi_class(mask_label,gender_label, age_label))#-
                self.indexes.append(cnt)#-
                self.groups.append(id)#-
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
        return image_transform, multi_class_label

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
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """

        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.all_labels})#-
        
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)#-
        train_index = train["idxs"].tolist()#-
        valid_index = valid["idxs"].tolist()#-
        
        return [Subset(self, train_index), Subset(self, valid_index)]
    
    def split_dataset_Kfold(self, k):
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.all_labels})#-
        
        val_ratio = frac(1, int(k)).limit_denominator()
        tmp_df = df.copy()
        fold_trn = []
        fold_val = []
        
        for i in range(1, int(str(frac(val_ratio).limit_denominator())[-1])):
            tmp_df, tmp_val = train_test_apart_stratify(tmp_df, group="groups", stratify="labels", test_size=val_ratio)#-

            tmp_trn = df.drop(df.index[tmp_val.index])
            fold_trn.append(tmp_trn["idxs"].tolist())
            fold_val.append(tmp_val["idxs"].tolist())

            val_ratio = float(frac(1, int(str(frac(val_ratio).limit_denominator())[-1])-1))
            if val_ratio == 1:
                tmp_val = tmp_df
                tmp_trn = df.drop(df.index[tmp_val.index])
                fold_trn.append(tmp_trn["idxs"].tolist())
                fold_val.append(tmp_val["idxs"].tolist())
                
        return [[Subset(self, train_index), Subset(self, valid_index)] for train_index, valid_index in zip(fold_trn, fold_val)]

class CustomDataset_2(Dataset):
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
        self.org_dir = '/opt/ml/input/data/train/images'
        self.aaf_dir = '/opt/ml/input/data2/images/aligned'
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
        
        self.target = target
        

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
        elif self.target == 'age' :
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
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'age' :
            self.classes = self.age_labels
        
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)#-
        train_index = train["idxs"].tolist()#-
        valid_index = valid["idxs"].tolist()#-
        
        return [Subset(self, train_index), Subset(self, valid_index)]
    
    def split_dataset_Kfold(self, k):
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'age' :
            self.classes = self.age_labels
            
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        val_ratio = frac(1, int(k)).limit_denominator()
        tmp_df = df.copy()
        fold_trn = []
        fold_val = []
        
        for i in range(1, int(str(frac(val_ratio).limit_denominator())[-1])):
            tmp_df, tmp_val = train_test_apart_stratify(tmp_df, group="groups", stratify="labels", test_size=val_ratio)#-

            tmp_trn = df.drop(df.index[tmp_val.index])
            fold_trn.append(tmp_trn["idxs"].tolist())
            fold_val.append(tmp_val["idxs"].tolist())

            val_ratio = float(frac(1, int(str(frac(val_ratio).limit_denominator())[-1])-1))
            if val_ratio == 1:
                tmp_val = tmp_df
                tmp_trn = df.drop(df.index[tmp_val.index])
                fold_trn.append(tmp_trn["idxs"].tolist())
                fold_val.append(tmp_val["idxs"].tolist())

        return [[Subset(self, train_index), Subset(self, valid_index)] for train_index, valid_index in zip(fold_trn, fold_val)]

class CustomDataset_3(Dataset):
    num_classes = 3 * 2 * 3

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    all_labels = []#-
    indexes = []#-
    groups = []#-

    def __init__(self, target, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.org_dir = '/opt/ml/input/data/train/images'
        self.aaf_dir = '/opt/ml/input/data2/images/aligned'
        self.test_dir = '/opt/ml/input/data/eval/images'
        self.test_data = pd.read_csv("csv/df_train_for_test.csv")
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()
        
        self.target = target
        

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
        elif self.target == 'age' :
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
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'age' :
            self.classes = self.age_labels
        
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        train, valid = train_test_apart_stratify(df, group="groups", stratify="labels", test_size=self.val_ratio)#-
        train_index = train["idxs"].tolist()#-
        valid_index = valid["idxs"].tolist()#-
        
        return [Subset(self, train_index), Subset(self, valid_index)]
    
    def split_dataset_Kfold(self, k):
        if self.target == 'mask' :
            self.classes = self.mask_labels
        elif self.target == 'gender' :
            self.classes = self.gender_labels
        elif self.target == 'age' :
            self.classes = self.age_labels
            
        df = pd.DataFrame({"idxs":self.indexes, "groups":self.groups, "labels":self.classes})#-
        
        val_ratio = frac(1, int(k)).limit_denominator()
        tmp_df = df.copy()
        fold_trn = []
        fold_val = []
        
        for i in range(1, int(str(frac(val_ratio).limit_denominator())[-1])):
            tmp_df, tmp_val = train_test_apart_stratify(tmp_df, group="groups", stratify="labels", test_size=val_ratio)#-

            tmp_trn = df.drop(df.index[tmp_val.index])
            fold_trn.append(tmp_trn["idxs"].tolist())
            fold_val.append(tmp_val["idxs"].tolist())

            val_ratio = float(frac(1, int(str(frac(val_ratio).limit_denominator())[-1])-1))
            if val_ratio == 1:
                tmp_val = tmp_df
                tmp_trn = df.drop(df.index[tmp_val.index])
                fold_trn.append(tmp_trn["idxs"].tolist())
                fold_val.append(tmp_val["idxs"].tolist())

        return [[Subset(self, train_index), Subset(self, valid_index)] for train_index, valid_index in zip(fold_trn, fold_val)]

class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


                    
    
class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

