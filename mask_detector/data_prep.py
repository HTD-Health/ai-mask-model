import os

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

DATASET_PATH = Path('./data/dataset')
CATEGORIES = {'no_mask': 0, 'mask': 1}


def verify_image(img_path):
    try:
        img = Image.open(img_path)
        img.verify()

        return True
    except IOError:
        return False


def prepare_data():
    dataset_list = []

    for category in CATEGORIES:
        path = DATASET_PATH/category
        mask = CATEGORIES.get(category)  # 0 - no_mask, 1 - mask

        for images_directory in tqdm([d for d in path.iterdir() if d.is_dir()]):
            for img_path in images_directory.iterdir():
                if verify_image(img_path):
                    dataset_list.append({
                        'image': str(img_path),
                        'mask': mask  # 0 - no_mask, 1 - mask
                    })

    return pd.DataFrame(dataset_list)


print('Preparing data...')
df = prepare_data()

print(f'Saving dataframe...')
df.to_csv('./data/dataframe/mask_df.csv', index=False)

print('Done')
