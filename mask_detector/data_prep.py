from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

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
    # TODO: add docstrings
    dataset_list = []

    for category in CATEGORIES:
        path = DATASET_PATH/category
        mask = CATEGORIES.get(category)  # 0 - no_mask, 1 - mask

        for images_directory in tqdm(list(path.iterdir())):
            for img_path in images_directory.iterdir():
                if verify_image(img_path):
                    dataset_list.append({
                        'image': str(img_path),
                        'mask': mask  # 0 - no_mask, 1 - mask
                    })

    return pd.DataFrame(dataset_list)


print('Preparing data...')
df = prepare_data()

print('Saving dataframe...')
df.to_csv('./data/dataframe/mask_df.csv', index=False)

print('Done')
