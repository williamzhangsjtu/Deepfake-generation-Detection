import torch
import torch.nn as nn 
import torch.nn.utils.rnn as rnn_utils

class LayerOut:
    
    def __init__(self, modules):
        self.Pooling = nn.AdaptiveAvgPool2d(1)
        self.features = {}

        self.hooks = [
            module.register_forward_hook(self.hook_fn)
            for module in modules
        ]

    def hook_fn(self, module, input, output):
        self.features[module] = self.Pooling(output).view(output.shape[0], -1)

    def remove(self):

        for hook in self.hooks:
            hook.remove()




class decoder(nn.Module):

    def __init__(self, inputdim):
        super(decoder, self).__init__()
        self.proj = nn.Linear(inputdim, 3 * 8 * 7 * 7)
        cnn_module = nn.ModuleList()
        channel = [24, 24, 12, 6, 6, 3]
        for i in range(len(channel) - 1):
            cnn_module.append(nn.ConvTranspose2d(channel[i], channel[i + 1], 4, 2, 1))
            cnn_module.append(nn.BatchNorm2d(channel[i + 1]))
            cnn_module.append(nn.ReLU())
        cnn_module[-1] = nn.Tanh()
        self.cnn_module = nn.Sequential(*cnn_module)


    def forward(self, input):
        proj = self.proj(input).view(input.shape[0], 3 * 8, 7, 7)

        return self.cnn_module(proj)#.squeeze(1)



class SimpleAttn(nn.Module):
    def __init__(self, dim):
        super(SimpleAttn, self).__init__()
        self.Linear = nn.Linear(dim, 1)
        self.Softmax = nn.Softmax(dim=1)
        nn.init.normal_(self.Linear.weight, 0, 0.1)
        nn.init.constant_(self.Linear.bias, 0.0)

    def forward(self, input, input_num=None):
        if (input_num is not None):
            idxs = torch.arange(input.shape[1]).repeat(input.shape[0]).view(input.shape[:2])
            masks = idxs.cpu() < input_num.cpu().view(-1, 1)
            masks = masks.to(torch.float).to(input.device)
            input = input * masks.unsqueeze(-1)
        
        alpha = self.Softmax(self.Linear(input))
        output = (alpha * input).sum(1)
        return output # B x D


class ModelInFrame(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True, **kwargs):
        super(ModelInFrame, self).__init__()
        self.DenseNet = DenseNet.features[:7]
        self.denseblocks = [
            #self.DenseNet.denseblock1,
            self.DenseNet.denseblock2,
            #self.DenseNet.denseblock3,
            #self.DenseNet.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad
        pretrain_dim = self.__getDenseDim()


        # )
        #attn_dim = kwargs['hidden_size'] * (2 if kwargs['bidirectional'] else 1)
        
        self.other = nn.ModuleDict({
            'reconstructor': decoder(pretrain_dim), 
            'linear1': nn.Linear(pretrain_dim, n_class),
            'Mapping': nn.Softmax(dim=1)
        })

        nn.init.normal_(self.other['linear1'].weight, 0, 0.1)
        nn.init.constant_(self.other['linear1'].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
    def get_other_param(self):

        return self.other.parameters()

    def get_param(self):
        
        
        return self.other.state_dict() if not self.Net_grad \
            else self.state_dict()
    
    def load_param(self, state_dict):
        
        if not self.Net_grad:
            self.other.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

    def forward(self, input, input_num=None):
        """
        input: B x C x W x H
        input_num: B
        """
        DenseOut = self._getDenseOut(input)

        dense_mapping = self.other['linear1'](DenseOut)
        #mapping_attn = dense_mapping

        return self.other['Mapping'](dense_mapping), \
            self.other['reconstructor'](DenseOut)  # B x C x W x H



class ModelAcrossFrame(nn.Module):
    

    def __init__(self, DenseNet, n_class, Net_grad=True, **kwargs):
        super(ModelAcrossFrame, self).__init__()
        self.DenseNet = DenseNet.features[:7]
        self.denseblocks = [
            self.DenseNet.denseblock1,
            self.DenseNet.denseblock2,
            #self.DenseNet.denseblock3,
            #self.DenseNet.denseblock4,
        ]
        self.BlockOut = LayerOut(self.denseblocks)
        self.Net_grad = Net_grad
        pretrain_dim = self.__getDenseDim()

        cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 5), padding=1, stride=(1, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 2)), 
            nn.ReLU(), nn.BatchNorm2d(4)
        )
        cnn_block2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 5), padding=1, stride=(2, 3)),
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 2)), 
            nn.ReLU(), nn.BatchNorm2d(16)
        )
        cnn_block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=1, stride=(2, 3)), 
            nn.AvgPool2d(kernel_size=(3, 5), padding=1, stride=(1, 2)), 
            nn.ReLU(), nn.BatchNorm2d(32)
        )
        attn_dim = kwargs['hidden_size'] * (2 if kwargs['bidirectional'] else 1)
        
        self.other = nn.ModuleDict({
            'cnn': nn.Sequential(
                cnn_block1, cnn_block2, cnn_block3, 
                nn.AdaptiveAvgPool2d(1)
            ),
            'lstm': nn.LSTM(pretrain_dim, batch_first=True, **kwargs), 
            'lstm_attn': SimpleAttn(attn_dim),
            'reconstructor': decoder(pretrain_dim), 
            'linear1': nn.Linear(32, n_class),
            'linear2': nn.Linear(attn_dim, n_class),
            'fusion_attn': SimpleAttn(n_class),
            'Mapping': nn.Softmax(dim=1)
        })

        nn.init.normal_(self.other['linear1'].weight, 0, 0.1)
        nn.init.constant_(self.other['linear1'].bias, 0)
        nn.init.normal_(self.other['linear2'].weight, 0, 0.1)
        nn.init.constant_(self.other['linear2'].bias, 0)
        
        

    def __getDenseDim(self):
        test = torch.zeros(1, 3, 224, 224)
        with torch.set_grad_enabled(False):
            self.DenseNet(test)
        dim = sum(self.BlockOut.features[block].shape[1] \
                for block in self.denseblocks)
        return dim

    def _getDenseOut(self, input):
        #B = input.shape[0]
        # with torch.set_grad_enabled(False):
        #     self.DenseNet(input)
        if (self.Net_grad):
            self.DenseNet(input)
        else:
            with torch.set_grad_enabled(False):
                self.DenseNet(input)

        output = torch.cat(list(self.BlockOut.features.values()), dim=1)
        return output  # B x D

    def get_Net_param(self):
        return self.DenseNet.parameters()
    
    def get_other_param(self):

        return self.other.parameters()

    def get_param(self):
        
        
        return self.other.state_dict() if not self.Net_grad \
            else self.state_dict()
    
    def load_param(self, state_dict):
        
        if not self.Net_grad:
            self.other.load_state_dict(state_dict)
        else:
            self.load_state_dict(state_dict)

    def forward(self, input, input_num=None):
        """
        input: B x T x C x W x H
        input_num: B
        """
        #device = input.device
        B, T = input.shape[:2]
        DenseOut = self._getDenseOut(input.view(-1, *input.shape[2:]))
        cnn_out = self.other['cnn'](DenseOut.view(B, T, -1).unsqueeze(1)).view(B, -1)
        fusion = self.other['linear1'](cnn_out)

        LSTM_IN = rnn_utils.pack_padded_sequence(
            DenseOut.view(B, T, -1), input_num, batch_first=True
        )
        LSTM_OUT, _ = self.other['lstm'](LSTM_IN)
        LSTM_OUT, _ = rnn_utils.pad_packed_sequence(LSTM_OUT, batch_first=True)
        lstm_attn = self.other['lstm_attn'](LSTM_OUT, input_num) # B x D
        
        series = self.other['linear2'](lstm_attn) # B x n_class
        #dense_mapping = self.other['linear1'](DenseOut).view(B, T, -1).sum(1) / (input_num.to(device).unsqueeze(-1)) # B x n_class


        mapping_attn = series + fusion
        
        #mapping_attn = attn_mapping

        #cnn_out = self.other['cnn'](DenseOut.view(B, T, -1).unsqueeze(1)).view(B, -1)
        return self.other['Mapping'](mapping_attn), \
            self.other['reconstructor'](DenseOut).view(*input.shape)  # B x T x C x W x H
