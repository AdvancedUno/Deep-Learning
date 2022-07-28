import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io


class CustomDataset(Dataset):
    def __init__(self,csv_file, root_dir, transform = None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))