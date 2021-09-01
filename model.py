import torch
import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_class):
        super(VGG, self).__init__()
        self.num_class = num_class
        self.vgg16 = models.vgg16_bn(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1000, out_features=64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=num_class)
        ) 
        
    def forward(self,x):
        out = self.vgg16(x)
        out = self.classifier(out)
        return out
    
class RESNET152(torch.nn.Module):
    def __init__(self,num_class):
        super(RESNET152,self).__init__()
        
        self.resnet152 = models.resnet152(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1000, out_features=64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=64, out_features=num_class)
        )
    def forward(self, x):
        out = self.resnet152(x)
        out = self.classifier(out)
        return out
    
class ViT(nn.Module):
    def __init__(self, num_class):
        super(ViT, self).__init__()
        
        self.vit = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=num_class)
    
    def forward(self, x):
        out = self.vit(x)
        return out
