import pandas as pd
import os

from pathlib import Path
from tqdm import tqdm

DATASET_PATH = Path('./data/dataset')
CATEGORIES = ['no_mask', 'mask']


def prepare_data():
    df = pd.DataFrame()

    for category in CATEGORIES:
        path = DATASET_PATH/category

        for directory in tqdm(list(path.iterdir())):
            for imgPath in directory.iterdir():
                df = df.append({
                    'image': str(imgPath),
                    'mask': CATEGORIES.index(category)  # 0 - no_mask, 1 - mask
                }, ignore_index=True)

    return df


print('Preparing data...')
df = prepare_data()

print(f'Saving dataframe...')
df.to_csv('./data/dataframe/mask_df.csv', index=False)

print('Done')
