from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate, Cutout, PadIfNeeded
from albumentations.augmentations.transforms import CoarseDropout
from albumentations.pytorch import ToTensor
import numpy as np


class album_compose():

    def __init__(self):
        self.albumentation_transforms = Compose([
            PadIfNeeded(min_height=36, min_width=36),
            Cutout(num_holes=4),
            # RandomSizedCrop':{'height':32,'width':32,'min_max_height':[28,28]},
            RandomCrop(32,32),
            HorizontalFlip(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
            ToTensor()])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img
