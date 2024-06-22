import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image

class FireAndNotFire(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        if not os.path.exists(img_path):
            print(f"File {img_path} not found.")
            return None, None
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image {img_path}")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image)  # Convert to PIL Image

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label
