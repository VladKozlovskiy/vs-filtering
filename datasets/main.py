import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import  Dataset


class PairedImageDatasetFromCSV(Dataset):
    def __init__(self, dataset_path):
        """
        :param dataset_dir: Путь к таблицам с метаданными (пути к файлам, ошибка регистрации, класс)
        """
        super().__init__()
        self.base_dir = "./"  # path to folder with data
        self.dataset_path = dataset_path
        self.annotations = pd.read_csv(self.dataset_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        fixed_path = os.path.join(self.base_dir, row['fixed_path'].removeprefix('filtration_imgs/'))
        warped_path = os.path.join(self.base_dir, row['warped_path'].removeprefix('filtration_imgs/'))
        dist, target = row['dist'], row['class']

        f_patch = np.array(Image.open(fixed_path).convert("RGB"))
        w_patch = np.array(Image.open(warped_path).convert("RGB"))

        return f_patch, w_patch, dist, target