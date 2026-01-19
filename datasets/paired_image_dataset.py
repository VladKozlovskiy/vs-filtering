import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import  Dataset
from preprocessors.paired_preprocessors import PairedCellImagePreprocessor


class PairedImageDatasetFromCSV(Dataset):
    def __init__(self, base_dir, annotation_path, label_column):
        super().__init__()
        self.base_dir = base_dir
        self.annotations = pd.read_csv(os.path.join(base_dir, annotation_path) )
        if label_column not in self.annotations.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in annotations. "
                f"Available columns: {list(self.annotations.columns)}"
            )
        self.label_column = label_column

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        fixed_path = os.path.join(self.base_dir, row['fixed_path'].removeprefix('filtration_imgs/'))
        warped_path = os.path.join(self.base_dir, row['warped_path'].removeprefix('filtration_imgs/'))
        target = row[self.label_column]

        f_patch = np.array(Image.open(fixed_path).convert("RGB"))
        w_patch = np.array(Image.open(warped_path).convert("RGB"))

        if hasattr(self, 'preprocessor'):
            f_patch, w_patch = self.preprocessor(f_patch, w_patch)
            return f_patch, w_patch, target
        return f_patch, w_patch, target

    def apply_preprocessor(self, preprocessor):
        if not isinstance(preprocessor, PairedCellImagePreprocessor):
            raise TypeError(
                "Expected PairedCellImagePreprocessor for paired dataset, "
                f"got {type(preprocessor).__name__}"
            )
        self.preprocessor = preprocessor
