from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate, Cutout
from albumentations.augmentations.transforms import CoarseDropout
from albumentations.pytorch import ToTensor
import numpy as np


class album_compose():

    def __init__(self):
        self.albumentation_transforms = Compose([
            Rotate((-7.0, 7.0)),
            Cutout(num_holes=4),
            CoarseDropout(),
            # RandomSizedCrop':{'height':32,'width':32,'min_max_height':[28,28]},
            # RandomCrop(10,10),
            HorizontalFlip(),
            Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ), ToTensor()])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img
