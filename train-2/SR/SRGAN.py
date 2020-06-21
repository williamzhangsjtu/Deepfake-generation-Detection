import math
import torch
import torchvision
import torch.nn as nn
import os
from SR.RRDB import RRDBNet
import logging
from collections import OrderedDict
import SR.lr_scheduler as lr_scheduler
from torch.nn.parallel import DataParallel, DistributedDataParallel

logger = logging.getLogger('base')

####################
# define network
####################

#### Discriminator
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


#### Define Network used for Perceptual Loss
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm

        model = torchvision.models.vgg19(pretrained=False)
        pre = torch.load(r'./train-2/checkpoint/vgg19-dcbb9e9d.pth')
        model.load_state_dict(pre)

        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def define_F(opt, use_bn=False):
    device = torch.device('cuda' if (not opt.no_cuda and torch.cuda.is_available()) else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF


# Define GAN loss
class GANLoss(nn.Module):
    def __init__(self, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_label(self, input, target_is_real):
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class SRGANModel:
    def __init__(self, opt):
        self.opt = opt
        # args.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if (not opt.no_cuda and torch.cuda.is_available()) else 'cpu')
        self.is_train = opt.is_train
        self.schedulers = []
        self.optimizers = []

        # define networks and load pretrained models
        self.netG = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23).to(self.device)
        self.netG = DataParallel(self.netG)

        if self.is_train:
            self.netD = Discriminator_VGG_128(in_nc=3, nf=64).to(self.device)
            self.netD = DataParallel(self.netD)
            self.netG.train()
            self.netD.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            self.cri_pix = nn.L1Loss().to(self.device)
            self.l_pix_w = 1e-2

            # G feature loss
            self.cri_fea = nn.L1Loss().to(self.device)
            self.l_fea_w = 1

            # load VGG perceptual loss
            self.netF = define_F(opt, use_bn=False).to(self.device)
            self.netF = DataParallel(self.netF)

            # GD gan loss - BCE LOSS
            self.cri_gan = GANLoss(1.0, 0.0).to(self.device)
            self.l_gan_w = 5e-3

            # # D_update_ratio and D_init_iters
            # self.D_update_ratio = 1
            # self.D_init_iters = 0

            # optimizers
            # G
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=1e-4,
                                                weight_decay=0,
                                                betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_G)
            # D
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=1e-4,
                                                weight_decay=0,
                                                betas=(0.9, 0.99))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            for optimizer in self.optimizers:
                # opt, lr_steps,
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, [50000, 100000, 200000, 300000], gamma=0.5))
            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            self.var_ref = data['GT'].to(self.device)

    def optimize_parameters(self):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        # pixel loss
        l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
        l_g_total += l_g_pix

        # feature loss
        real_fea = self.netF(self.var_H).detach()
        fake_fea = self.netF(self.fake_H)
        l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
        l_g_total += l_g_fea

        pred_g_fake = self.netD(self.fake_H)
        pred_d_real = self.netD(self.var_ref).detach()
        l_g_gan = self.l_gan_w * (
            self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
            self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_g_total += l_g_gan

        l_g_total.backward()
        self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G

        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
        l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_fea'] = l_g_fea.item()
        self.log_dict['l_g_gan'] = l_g_gan.item()
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def save_network(self, network, network_label, iter_label, path):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save(self, iter_step, path):
        self.save_network(self.netG, 'G', iter_step, path)
        self.save_network(self.netD, 'D', iter_step, path)

    def save_training_state(self, epoch, iter_step, path):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        if not os.path.exists(path):
            os.makedirs(path)
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(path, save_filename)
        torch.save(state, save_path)


