
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MyTrainDataset(Dataset) :
    def __init__(self, path, transform, train=True):
        # 데이터와 경로 이미지 받아오기
        self.img_data = pd.read_csv(path['trainCsv'])
        self.img_dir = path['image']
        
        # 라벨 리스트
        self.label = [self.img_data['gender'],
                     self.img_data['age'],
                     self.img_data['mask']]
        
        self.label = pd.DataFrame(self.label)
        
        # 각 feature별 클래스 생성
        self.gen_classes = ['male', 'female']
        self.age_classes = [str(i) for i in range(18,61)]
        self.mask_classes = ['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal']
        
        # dataloader 특성 받기
        self.train = train
        self.transform = transform
        self.path = path
        self._repr_indent = 4
        
    def __len__(self) :
        return len(self.img_data)
    
    def __getitem__(self, idx) :
        # 이미지 경로 받아서 입력
        person_path = os.path.join(self.img_dir, self.img_data.iloc[idx,4])
        img_path = os.path.join(person_path, 'mask2.jpg')
        image = Image.open(img_path)
        
        # 이미지에 transform이 있다면 실행
        if self.transform :
            image = self.transform(image)
        
        # 라벨 입력
        label = self.label.iloc[:, idx]

        return image, label
    
    def __repr__(self):
        '''
        https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py
        '''
        head = "(Inform) My Custom Dataset"
        data_path = self._repr_indent*" " + "Data path: {}".format(self.path['image'])
        label_path = self._repr_indent*" " + "Label path: {}".format(self.path['trainCsv'])
        num_data = self._repr_indent*" " + "Number of datapoints: {}".format(self.__len__())
        num_classes = self._repr_indent*" " + "Number of gender classes: {}".format(len(self.gen_classes))

        return '\n'.join([head,
                          data_path, label_path, 
                          num_data, num_classes])