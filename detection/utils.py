from torchvision import transforms
import torchvision.transforms.functional as F
import logging, os, sys
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import random
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image, ImageFilter

class RotationTransform:
    def __init__(self, angles = [-180, -90, 0, 90, 180]):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)


class RandomMask:
    def __init__(self, p=0.1):
        self.p = p
        self.var_limit = [0.0, 5.0]
        self.mean = 0

    def __call__(self, img):
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        gauss = torch.normal(self.mean, var ** 2, (3, 224, 224))
        mask = torch.randint(0, 10, (3, 224, 224)) >= 1
        return mask.to(float) * gauss.to(float) + img.to(float)


class GaussianBlur:
    def __init__(self, radius = 2):
        self.radius = radius

    def __call__(self, img):
        img2 = img.filter(ImageFilter.GaussianBlur(radius = self.radius)) 
        return img2

class GaussianNoise:
    def __init__(self, var_limit = [10.0, 50.0]):
        self.mean = 0
        self.var_limit = var_limit
        
    def __call__(self, img):
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        gauss = torch.normal(self.mean, var ** 2, (3, 224, 224))

        return img + gauss


def train_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        RotationTransform(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def test_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def simple_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def genlogger(outputfile):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(logging.INFO)
    stdlog = logging.StreamHandler(sys.stdout)
    stdlog.setFormatter(formatter)
    file_handler = logging.FileHandler(outputfile)
    file_handler.setFormatter(formatter)
    # Log to stdout
    logger.addHandler(file_handler)
    logger.addHandler(stdlog)
    return logger




def evaluate_sample(model, dataloader, device, threshold=0.5):

    outputs, labels = [], []
    with torch.set_grad_enabled(False):
        for idx, (imgs, label, nums, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            label = label.numpy()
            output = model(imgs, nums)[0].cpu().numpy()
            
            for i in range(len(output)):
                prob = output[i] - threshold
                pred = prob >= (0 if np.max(prob) > 0 else np.max(prob))
                outputs.append(pred.astype(int))
                labels.append(label[i])
    
    f1_macro = f1_score(np.stack(labels), np.stack(outputs), average='macro')
    f1_micro = f1_score(np.stack(labels), np.stack(outputs), average='micro')
    return f1_macro, f1_micro









def evaluate(model, dataloader, device, criterion=None, threshold=0.5):
    index_dict, label_dict, index_len = {}, {}, {}
    loss_mean = 0
    model = model.eval()
    count = 0.1
    with torch.set_grad_enabled(False):
        for imgs, labels, nums, indexes in tqdm(dataloader):
            count += 1
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs, nums)
            if criterion is not None:
                loss = criterion(output, imgs, labels)
                loss_mean += loss.cpu().item()
            
            if (output.__class__ == tuple):
                output = output[0].cpu().numpy()
                
            else:
                output = output.cpu().numpy()
            nums = nums.numpy()
            labels = labels.cpu().numpy()

            for i in range(len(output)):
                
                index_dict[indexes[i]] = index_dict.setdefault(
                    indexes[i], np.zeros(output.shape[1:])) + output[i] * nums[i]
                label_dict[indexes[i]] = labels[i]
                index_len[indexes[i]] = index_len.setdefault(
                    indexes[i], 0) + nums[i]
    outputs, labels, probs = [], [], []
    for (index, prob) in index_dict.items():
        pred = np.argmax(prob)
        outputs.append(pred.astype(int))
        labels.append(label_dict[index])
        probs.append(np.max(prob) / index_len[index])
    f1_macro = f1_score(labels, outputs, average='macro')
    f1_micro = f1_score(labels, outputs, average='micro')
    acc = accuracy_score(labels, outputs)
    auc = roc_auc_score(labels, probs, average='macro')
    return loss_mean / count, f1_macro, f1_micro, acc, auc

