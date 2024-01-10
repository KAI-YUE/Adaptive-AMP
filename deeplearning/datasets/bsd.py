import os
import pickle
from PIL import Image

from torchvision import datasets, transforms
from torch import tensor, long

from torch.utils.data import Dataset, DataLoader

def BSD(data_path):
    channel = 1
    # channel = 3
    im_size = (64, 64)
    num_classes = 1
    mean = 0.4975
    std = 0.2045
    
    with open(os.path.join(data_path, "datapair.dat"), "rb") as fp:
        record = pickle.load(fp)
    
    transform = transforms.Compose([
                                    transforms.Resize(im_size),
                                    transforms.ToTensor(), 
                                    # transforms.Normalize(mean=mean, std=std),
                                    ])
    
    datapair = record["data_pair"]
    root_dir = record["root"]
    dataset = ImageDataset(root_dir, datapair, transform)

    properties = {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "n_train": len(dataset),
        "dst_train": dataset,
        "dst_test": dataset,
        "ram_load": False,
        "mean": mean,
        "std": std,
    }

    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties


class ImageDataset(Dataset):
    def __init__(self, root_dir, datapair, transform):
        self.root_dir = root_dir
        self.datapair = datapair
        self.transform = transform

    def __getitem__(self, index):
        
        img_name = os.path.join(self.root_dir, self.datapair[index][0], self.datapair[index][1])
        # img = Image.open(img_name).convert('RGB')
        img = Image.open(img_name)

        img = self.transform(img)
        return img, tensor(self.datapair[index][2], dtype=long)

    def __len__(self):
        return len(self.datapair)