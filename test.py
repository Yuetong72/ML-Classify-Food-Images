# -*- coding: utf-8 -*-
import torch,os
import numpy as np
import cv2 as cv
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from AI.utils import *
from AI.efficientnet.model import EfficientNet
def get_dirs_files(path):
    files=[]
    dirs_=[]
    dirs_.append(path)
    dirs=[]
    while len(dirs_)!=0:
        dir=dirs_.pop()
        for line in os.listdir(dir):
            c=os.path.join(dir,line)
            if os.path.isdir(c):
                dirs_.append(c)
                dirs.append(c)
            else:
                files.append(c)
    return dirs,files

class Classifier:
    def __init__(self,model,transform,device):
        self.model=model
        self.transform=transform
        self.device=device

    def predict(self,image):
        if not isinstance(image,Image.Image):
            image = Image.fromarray(image)
        img_tran = self.transform(image)
        img_tran = img_tran.unsqueeze(0)

        with torch.no_grad():
            img_tensor = img_tran.to(self.device)
            logits = self.model(img_tensor)
            if isinstance(logits,tuple):
                logits=logits[0]
            # predict=logits.argmax(1).item()
            pred_top3 = logits.argsort(dim=1)[:, -3:].cpu().numpy()
        return pred_top3[0]


if __name__=="__main__":
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform= transforms.Compose([transforms.Resize([224 ,224],3),
                                 transforms.ToTensor()])

    model = EfficientNet.from_name("efficientnet-b2", override_params={'num_classes': 251})
    state_dict = torch.load("results_classification/weights_best/epoch_1_error_0.1598_params.pth")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    classifier=Classifier(model,transform,device)

    test_dir="./data/val_set"
    fid=open("result.csv",'w')
    count = 0
    for line in os.listdir(test_dir):
        if count % 100 == 0:
            print(count)

        img_path=os.path.join(test_dir,line)

        img=Image.open(img_path).convert("RGB")
        y_pred=classifier.predict(img)
        fid.write(img_path[-15:]+",%d %d %d\n"%(y_pred[0],y_pred[1],y_pred[2]))
        count+=1
