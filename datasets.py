from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import random


class UKTDataset(Dataset):

    def __init__(self, data_path: str = "./archive/UKTFace/",
                    limit=None,
                    image_height: int = 512,
                    image_width: int = 512):

        super(UKTDataset, self).__init__()

        self.filenames = os.listdir(data_path)
        self.filenames = [f for f in self.filenames if len(f.split('_'))==4]
        random.shuffle(self.filenames)

        if limit is not None:
            self.filenames = self.filenames[0:limit]

        self.age = [int(f.split('_')[0]) for f in self.filenames]
        self.sex = [int(f.split('_')[1]) for f in self.filenames]
        self.ethnicity = [int(f.split('_')[2]) for f in self.filenames]

        X = []
        for f in tqdm(self.filenames, desc="Loading UKT dataset to memory"):
            filename = os.path.join(data_path, f)
            image = Image.open(filename)
            # resize image
            image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
            # turn to numpy array
            data = np.asarray(image)
            # min-max scale
            data = (data.astype('float') - data.min())/(data.max() - data.min())
            torchify = torch.FloatTensor(data).unsqueeze(0)
            X.append(torchify)
        
        self.X = torch.cat(X, dim=0)
        self.X = self.X.mean(dim=3).unsqueeze(-1)
        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]
