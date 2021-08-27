import torch
import torch.nn as nn

class NaiveCNN(nn.Module):

    def __init__(self, config=None):

        super().__init__()
        self.config = config

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):

        return self.feature_extractor(x)


if __name__=="__main__":

    model = NaiveCNN()
    print(model(torch.zeros(2, 3, 512, 384)))