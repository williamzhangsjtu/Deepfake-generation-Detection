import torch, torchvision
from torch import optim
import losses
from torchvision import transforms
import pandas as pd
import numpy as np 
import yaml
import argparse, os
import time
from dataloader import get_dataloader
import model as M
from tqdm import tqdm
import utils


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config/config.yaml', type=str)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-t', '--test', action='store_true')
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-p', '--path', default=None, type=str)
parser.add_argument('-f', '--file', default='model.th', type=str)

args = parser.parse_args()
#device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)



def get_config(config):
    with open(config) as c:
        data = yaml.safe_load(c)
    return data

def one_epoch(model, optimizer, criterion, dataloader, iftrain=True, grad_clip=None):
    loss_mean = 0
    if (iftrain): model.train()
    else: model.eval()
    count = 0

    for images, labels, nums, _ in tqdm(dataloader):
        count += 1
        images = images.to(device)
        labels = labels.to(device)
        nums = nums.to(device)
        with torch.set_grad_enabled(iftrain):
            outputs = model(images, nums)
            loss = criterion(outputs, images, labels)
            #loss = criterion(outputs, labels)
            loss_mean += loss.cpu().item()
            if (iftrain):
               optimizer.zero_grad()
               loss.backward()
               if grad_clip:
                   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
               optimizer.step() 
    return loss_mean / count


    

def run(config_file):
    config = get_config(config_file)

    cur_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    outdir = os.path.join(config['outputdir'], cur_time)
    os.makedirs(outdir)

    logger = utils.genlogger(os.path.join(outdir, 'log.txt'))
    logger.info("Output Path: {}".format(outdir))
    logger.info("<---- config details ---->")
    for key in config:
        logger.info("{}: {}".format(key, config[key]))
    logger.info("<---- end of config ---->")


    ##########<------------------- Data ----------------->##########
    train_df = pd.read_csv(config['train_label'], sep=',')
    dev_df = pd.read_csv(config['dev_label'], sep=',')
    n_class = config['n_class']

    train_dict = {index:label for index, label in train_df.values}
    dev_dict = {index:label for index, label in dev_df.values}
    #print(dev_dict)

    train_transform = utils.train_transform()
    dev_transform = utils.test_transform()
    train_dataloader = get_dataloader(
        config['train_data'], train_dict, train_transform, \
        config['time_step'], **config['dataloader_param']
    )
    dev_dataloader = get_dataloader(
        config['dev_data'], dev_dict, dev_transform, \
        config['time_step'], **config['dataloader_param']
    )
    ##########<------------------- Data ----------------->##########




    ##########<------------------- Model ----------------->##########
    Net = torchvision.models.densenet201(pretrained=config['pretrain'])
    model = getattr(M, config['model'])(
        Net, n_class=n_class, **config['model_param']
    )
    logger.info("model: {}".format(str(model)))
    origin_model = model
    # if (torch.cuda.device_count() > 1):
    #     model = torch.nn.DataParallel(model)
    # logger.info("Use {} GPU(s)".format(torch.cuda.device_count()))
    model = model.to(device)
    ##########<------------------- Model ----------------->##########



    ##########<------------------- Optimizer ----------------->##########
    optimizer = getattr(optim, config['optim'])(
        origin_model.parameters(), lr=config['lr']
    )
    ##########<------------------- Optimizer ----------------->##########


    lr_scheduler = getattr(optim.lr_scheduler, config['lr_scheduler'])(
        optimizer, **config['scheduler_param']
    )
    criterion = getattr(losses, config['Loss'])()#lambd=0.5)

    



    

    dev_loss, f1_macro, f1_micro, acc, auc = utils.evaluate(
        model, dev_dataloader, device, criterion, config['threshold'])
    best_f1 = f1_macro + f1_micro
    logger.info("dev_loss: {:.4f}\tf1_macro: {:.4f}\tf1_micro: {:.4f}\tacc: {:.4f}\tauc: {:.4f}"\
        .format(dev_loss, f1_macro, f1_micro, acc, auc))
 
    for epoch in range(1, config['n_epoch'] + 1):
        logger.info("<---- Epoch: {} start ---->".format(epoch))

        train_loss = one_epoch(
            model, optimizer, criterion, \
            train_dataloader, True, config['grad_clip']
        )

        dev_loss, f1_macro, f1_micro, acc, auc = utils.evaluate(
            model, dev_dataloader, device, criterion, config['threshold'])


        logger.info("train_loss: {:.4f}\tdev_loss: {:.4f}".format(train_loss, dev_loss))

        

        logger.info("DEV: f1_macro: {:.4f}\tf1_micro: {:.4f}\tacc: {:.4f}\tauc: {:.4f}".format(f1_macro, f1_micro, acc, auc))


        if epoch % config['saveinterval'] == 0:
            model_path = os.path.join(outdir, 'model_{}.th'.format(epoch))
            torch.save({
                "param": origin_model.get_param(),
                "config": config
            }, model_path)

        if best_f1 < f1_macro + f1_micro:
            model_path = os.path.join(outdir, 'model_acc.th')
            torch.save({
                "param": origin_model.get_param(),
                "config": config
            }, model_path)
            best_f1 = f1_macro + f1_micro


        schedarg = dev_loss if lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau' else None
        lr_scheduler.step(schedarg)


def evaluate(path, file):


    logger = utils.genlogger(os.path.join(path, 'stats.txt'))
    logger.info("Output Path: {}".format(path))
    logger.info("<---- Evaluation on Test Set ---->")

    obj = torch.load(os.path.join(path, file), lambda stg, loc: stg)
    config = obj['config']
    dev_df = pd.read_csv(config['dev_label'], sep=',')

    test_label = {index:label for index, label in dev_df.values}
    #test_label = obj['test_label']
    

    Net = torchvision.models.densenet201(pretrained=config['pretrain'])

    model = getattr(M, config['model'])(
        Net, n_class=config['n_class']
    )
    model.load_param(obj['param'])
    model = model.eval().to(device)

    test_transform = utils.test_transform()
    
    test_dataloader = get_dataloader(
        config['dev_data'], test_label, test_transform, \
        T = config['time_step'], **config['dataloader_param']
    )

    _, f1_macro, f1_micro, acc, auc = utils.evaluate(
        model, test_dataloader, device, None, config['threshold'])

    logger.info("<---- test evaluation: ---->")
    logger.info("f1_macro: {:.4f}\tf1_micro: {:.4f}\tacc: {:.4f}\tauc: {:.4f}".format(f1_macro, f1_micro, acc, auc))



if __name__ == '__main__':
    if not args.test:
        run(args.config)
    else:
        evaluate(args.path, args.file)
