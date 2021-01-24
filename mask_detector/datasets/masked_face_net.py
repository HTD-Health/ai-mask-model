""" Dataset module
"""
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToPILImage, ToTensor


class MaskedFaceNetDataset(Dataset):
    def __init__(self, csv_file, image_size):
        self.dataFrame = pd.read_csv(csv_file)

        self.transform = Compose([
            Resize(image_size),  # for MobileNetV2 - set image size to 256
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]

        mask = torch.tensor([row['mask']], dtype=torch.float)
        image = Image.open(row['image'])

        return self.transform(image), mask

    def __len__(self):
        return len(self.dataFrame.index)
