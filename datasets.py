from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torch


class UKTDataset(Dataset):

    def __init__(self, data_path: str = "./archive/UKTFace/",
                    image_height: int = 512,
                    image_width: int = 512):

        super(UKTDataset, self).__init__()

        X = []
        for f in tqdm(os.listdir(data_path)[0:10000], desc="Loading UKT dataset to memory"):
            filename = os.path.join(data_path, f)
            image = Image.open(filename)
            # resize image
            image = image.resize((image_width,image_height), Image.Resampling.LANCZOS)
            # turn to numpy array
            data = np.asarray(image)
            # min-max scale
            data = (data.astype('float') - data.min())/(data.max() - data.min())
            torchify = torch.FloatTensor(data).unsqueeze(0)
            X.append(torchify)
        
        self.X = torch.cat(X, dim=0)
        print(self.X.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]
