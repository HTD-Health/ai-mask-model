""" Dataset module
"""
import cv2
import numpy as np
import pandas as pd
from torch import float, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


class MaskDataset(Dataset):
    def __init__(self, csv_file, image_size=100):
        self.dataFrame = pd.read_csv(csv_file)

        self.transform = Compose([
            ToPILImage(),
            Resize((image_size, image_size)),
            ToTensor(),  # [0, 1] | [no_mask, mask]
        ])

    def __getitem__(self, key):
        row = self.dataFrame.iloc[key]

        mask = tensor([row['mask']], dtype=float)
        image = cv2.imdecode(np.fromfile(
            row['image'], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        return self.transform(image), mask

    def __len__(self):
        return len(self.dataFrame.index)
