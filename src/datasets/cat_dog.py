import mlconfig
import torch
from torch.utils import data
from torchvision import transforms
import os
import numpy as np
from PIL import Image


@mlconfig.register
class Cat_Dog(data.DataLoader):

    def __init__(self, root: str, list_file: str, batch_size: int, **kwargs):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        dataset = ListDataset(root, list_file=list_file, transform=transform)

        super(Cat_Dog, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)

        
class ListDataset(data.Dataset):

    def __init__(self, root, list_file, transform):
        self.root = os.path.join(root, list_file)
        self.fnames = os.listdir(self.root)
        self.transform = transform
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        img = self.transform(img)
        if 'cat' in fname:
            target = 0
        else:
            target = 1
        return img, target


    def __len__(self):
        return self.num_samples