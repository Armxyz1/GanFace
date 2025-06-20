import pandas as pd
import os
import shutil

train = pd.read_csv('dataset/train.csv')

os.makedirs('dataset/train_images', exist_ok=True)

count = 0
for idx, row in train.iterrows():
    if row.race == 'Black':
        image_path = f"dataset/{row.file}"
        if os.path.exists(image_path):
            shutil.copy(image_path, f'dataset/train_images/{count}.jpg')
            count += 1


val = pd.read_csv('dataset/val.csv')

os.makedirs('dataset/val_images', exist_ok=True)

count = 0
for idx, row in val.iterrows():
    if row.race == 'Black':
        image_path = f"dataset/{row.file}"
        if os.path.exists(image_path):
            shutil.copy(image_path, f'dataset/val_images/{count}.jpg')
            count += 1