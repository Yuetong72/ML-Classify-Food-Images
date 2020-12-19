import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import cv2 as cv
import os


def pil_loader(path):
    img = Image.open(path)
    img = img.convert('RGB')
    return img

class ImageList(data.Dataset):

    def read_txt(self,txt_file):
        data=[]
        with open(txt_file) as fid:
            for line in fid:
                line=line.strip().split(",")
                label=int(line[1])
                data.append([line[0],label])
        return data

    def __init__(self,root, txt_file, transform=None, loader=pil_loader,image_mode="RGB"):
        self.data=self.read_txt(txt_file)
        self.transform = transform
        self.loader = loader
        self.image_mode=image_mode
        self.root=root

    def __getitem__(self, index):
        img_path,label = self.data[index]
        label=np.array(label,np.int32)
        label=torch.from_numpy(label)
        if self.root is None:
            img = self.loader(img_path)
        else:
            img = self.loader(os.path.join(self.root,img_path))
 
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

