from torch.utils.data import Dataset, DataLoader
import argparse
# import options as option
import os
import cv2
import torch
from torch.autograd import Variable
import logging

from models import Autoencoder
from image_augmentation import random_transform
from image_augmentation import random_warp
from SR.SRGAN import SRGANModel


class ESRDataset(Dataset):
    def __init__(self, dirpath, modelpath):
        self.cage = [x.path for x in os.scandir(os.path.join(dirpath, 'cage')) if x.name.endswith('jpg') or x.name.endswith('.png')]
        self.trump = [x.path for x in os.scandir(os.path.join(dirpath, 'trump')) if x.name.endswith('jpg') or x.name.endswith('.png')]
        self.real_dir = self.cage + self.trump

        # auto encoder initialization, generate low quality image
        checkpoint = torch.load(modelpath, map_location='cpu')
        self.model = Autoencoder()
        self.model.load_state_dict(checkpoint['state'])
        self.random_transform_args = {
            'rotation_range': 10,
            'zoom_range': 0.05,
            'shift_range': 0.05,
            'random_flip': 0.4,
        }

    def __getitem__(self, index, tp=True):
        # return 256*256 image
        cls = 'A' if index < len(self.cage) else 'B'
        img = cv2.resize(cv2.imread(self.real_dir[index]), (256, 256))/255.0

        img, target_img = random_warp(random_transform(img, **self.random_transform_args))
        img = self.model(Variable(torch.unsqueeze(self.toTensor(img).float(), 0)), cls)

        img = torch.squeeze(img, 0)
        target_img = self.toTensor(target_img)
        target_img = target_img.float()
        return {'LQ': img, 'GT': target_img}

    def __len__(self):
        return len(self.real_dir)

    def toTensor(self, img):
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img


def train(args):
    # initialization
    logger = logging.getLogger('base')
    train_dataset = ESRDataset(dirpath='./train-2/data', modelpath='./train-2/checkpoint/autoencoder.t7')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    current_step = 0
    start_epoch = 0
    save_paths = args.path
    model = SRGANModel(args)

    tot_step = 400000
    tot_epochs = tot_step * args.batch_size / len(train_dataset)

    for epoch in range(start_epoch, int(tot_epochs)):
        for idx, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > tot_step:
                break

            #### update learning rate
            for scheduler in model.schedulers:
                scheduler.step(current_step)

            #### training
            model.feed_data(train_data)
            model.optimize_parameters()

            #### log , save models and training states
            if current_step % 100 == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)
                logger.info('Saving models and training states.')
                model.save(current_step, save_paths)
                model.save_training_state(epoch, current_step, save_paths)

    logger.info('Saving the final model.')
    model.save('latest', save_paths)
    logger.info('End of training.')


if __name__ == "__main__":
    #### options
    parser = argparse.ArgumentParser(description='ESRGAN-Pytorch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--is-train', default=True, help='training or not')
    parser.add_argument('--logInterval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--path', type=str, default='./train-2/checkpoint/models')
    args = parser.parse_args()
    train(args)