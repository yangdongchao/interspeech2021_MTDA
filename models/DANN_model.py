import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import os

from progressbar import ProgressBar
import torch.optim.lr_scheduler as lr_scheduler
from prettytable import PrettyTable

from PIL import Image
from matplotlib.cm import get_cmap
from tensorboardX import SummaryWriter as Logger
def nll_loss(output, target):
    '''Negative likelihood loss. The output should be obtained using F.log_softmax(x). 
    
    Args:
      output: (N, classes_num)
      target: (N, classes_num)
    '''
    loss = - torch.mean(target * output)
    return loss
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device='cuda'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
class  nnMeanAndMax(nn.Module):
    def __init__(self):
        super(nnMeanAndMax,self).__init__()
    def forward(self,x):
        x = torch.mean(x,dim=-1)
        (x, _) = torch.max(x,dim=-1)
        return x
class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)  # 去掉维数为1的的维度
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc(x)
        
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output

class EncoderCNN(nn.Module):
    def __init__(self, classes_num, activation):
        super(EncoderCNN, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.feature = None
        self.target = None

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
    def init_weights(self):
        init_layer(self.fc)
    def forward(self, x, u):

        """
        :param x: (batch_size, times_steps, freq_bins)
        :param u: (batch_size,)
        :return:
        """
        # print('x ',x.shape)
        # print('u ',u.shape)
        tmp_y = u
        u_new = torch.reshape(u,(u.shape[0],1,1))
        # print('u_new ',u_new.shape)
        u_new = u_new.repeat((1,x.shape[1],1))
        # print('u_new2 ',u_new.shape)
        x = torch.cat((x,u_new),dim=-1) # (b,1,t,f+1)
        # print('x0 ',x.shape)
        x = x[:,None,:,:]
        # print('x1 ',x.shape)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') 
        # print('x_conv1 ',x.shape)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x_conv2 ',x.shape)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # (batch_size, feature_maps, time_steps, freq_bins+1)
        E = x
        # print('x_cov3 ',x.shape)
        z = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        # print('z ',z.shape)
        z = torch.mean(z, dim=3)        # (batch_size, feature_maps, time_stpes)
        # print('z_mean ',z.shape)
        (z, _) = torch.max(z, dim=2)    # (batch_size, feature_maps)
        # print('z_max ',z.shape)
        z = self.fc(z)
        # print('z_fc ',z.shape)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(z, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(z)
        # print('output ',output.shape)
        self.feature = E.detach()
        self.target = tmp_y.detach()
        return output, x
class EncoderCNNWithoutIndex(nn.Module):
    def __init__(self, classes_num, activation):
        super(EncoderCNNWithoutIndex, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.feature = None
        self.target = None

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
    def init_weights(self):
        init_layer(self.fc)
    def forward(self, x, u):

        """
        :param x: (batch_size, times_steps, freq_bins)
        :param u: (batch_size,)
        :return:
        """
        # print('x ',x.shape)
        # print('u ',u.shape)
        tmp_y = u
        # u_new = torch.reshape(u,(u.shape[0],1,1))
        # # print('u_new ',u_new.shape)
        # u_new = u_new.repeat((1,x.shape[1],1))
        # print('u_new2 ',u_new.shape)
        #x = torch.cat((x,u_new),dim=-1) # (b,1,t,f+1)
        # print('x0 ',x.shape)
        x = x[:,None,:,:]
        # print('x1 ',x.shape)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') 
        # print('x_conv1 ',x.shape)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x_conv2 ',x.shape)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # (batch_size, feature_maps, time_steps, freq_bins+1)
        E = x
        # print('x_cov3 ',x.shape)
        z = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        # print('z ',z.shape)
        z = torch.mean(z, dim=3)        # (batch_size, feature_maps, time_stpes)
        # print('z_mean ',z.shape)
        (z, _) = torch.max(z, dim=2)    # (batch_size, feature_maps)
        # print('z_max ',z.shape)
        z = self.fc(z)
        # print('z_fc ',z.shape)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(z, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(z)
        # print('output ',output.shape)
        self.feature = E.detach()
        self.target = tmp_y.detach()
        return output, x
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        #self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc1 = nn.Linear(256,128)
        self.fc = nn.Linear(128, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x,u):
        tmp_y = u
        x = x[:,None,:,:]
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        z = self.layer3(out)
        #z = self.layer4(out)
        E = z
        #print('E ',E.shape)
        #clf = F
        #print('clf ',clf.shape)
        # clf = F.avg_pool2d(z, 4)
        # print('clf2 ',clf.shape)
        # clf = out.view(clf.size(0), -1)
        # print('clf3 ',clf.shape)
        z_x = torch.mean(z, dim=3)        # (batch_size, feature_maps, time_stpes)
        # print('z_mean ',z.shape)
        (z_x, _) = torch.max(z_x, dim=2)    # (batch_size, feature_maps)
        #print('z_x ',z_x.shape)
        out = self.fc1(z_x)
        out = self.fc(out)
        output = F.log_softmax(out, dim=-1) # 求log_softmax

        self.feature = E.detach()  # 保存
        self.target = tmp_y.detach()

        return output, z


def ResNet18():
    return ResNet(ResidualBlock)


class Encoder2019(nn.Module):
    def __init__(self, classes_num, activation):
        super(Encoder2019, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.feature = None
        self.target = None

        self.fc = nn.Linear(256, 100, bias=True)
        self.fc1 = nn.Linear(100,classes_num)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(10)
        self.init_weights()
    def init_weights(self):
        init_layer(self.fc)
    def forward(self, x, u):

        """
        :param x: (batch_size, times_steps, freq_bins)
        :param u: (batch_size,)
        :return:
        """
        # print('x ',x.shape)
        # print('u ',u.shape)
        tmp_y = u
        # u_new = torch.reshape(u,(u.shape[0],1,1))
        # # print('u_new ',u_new.shape)
        # u_new = u_new.repeat((1,x.shape[1],1))
        # print('u_new2 ',u_new.shape)
        #x = torch.cat((x,u_new),dim=-1) # (b,1,t,f+1)
        # print('x0 ',x.shape)
        x = x[:,None,:,:]
        # print('x1 ',x.shape)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') 
        # print('x_conv1 ',x.shape)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        # print('x_conv2 ',x.shape)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # (batch_size, feature_maps, time_steps, freq_bins+1)
        
        # print('x_cov3 ',x.shape)
        #z = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        # print('z ',z.shape)
        #print('x ',x.shape)
        #E = x
        z = F.avg_pool2d(x, kernel_size=(1,1))
        E = z
        #print('z ',z.shape)
        z = torch.mean(z, dim=3)        # (batch_size, feature_maps, time_stpes)
        # print('z_mean ',z.shape)
        (z, _) = torch.max(z, dim=2)    # (batch_size, feature_maps)
        Encoder = z
        # print('z_max ',z.shape)
        z = self.fc(z)  # 256--->100
        z = self.bn1(z)
        z = F.relu_(z)
        # 第二层 fc
        z = self.fc1(z) # 100->10
        z = self.bn2(z)
        z = F.relu_(z)
        # print('z_fc ',z.shape)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(z, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(z)
        # print('output ',output.shape)
        self.feature = E.detach()
        self.target = tmp_y.detach()
        return output, x
class DiscConv2019(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv2019, self).__init__()
        self.fc = nn.Linear(256,1)
        self.bn = nn.BatchNorm1d(1)
        # print(self.net)

    def forward(self, x):
        # print('x_DiscConv ',x.shape)
        z = self.fc(x)
        z = self.bn(z)
        z = F.relu_(z)
        #z = F.softmax(z)
        return z
class DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnMeanAndMax(),
            nn.Linear(nh, nout),
        )
        # print(self.net)

    def forward(self, x):
        # print('x_DiscConv ',x.shape)
        return self.net(x)
class DiscConv_oneHot(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv_oneHot, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnMeanAndMax(),
            nn.Linear(nh, nout),
        )
        # print(self.net)

    def forward(self, x):
        # print('x_DiscConv ',x.shape)
        z = self.net(x)
        output = F.softmax(z)
        return output
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.tsne = TSNE(n_components=2)
        self.pca = PCA(n_components=2)

        self.train_log = opt.outf + '/train_gan2_new_data_dict_without_index.log'
        self.model_path = opt.outf + '/model_gan2_new_data_dict_without_index.pth'
        self.logger = Logger(opt.outf)
        self.best_acc_tgt = 0

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        try:
            print('load model from {}'.format(self.model_path))
            self.load_state_dict(torch.load(self.model_path))
            print('done!')
        except:
            print('failed!')

    def acc_reset_mnist(self):
        self.hit_domain, self.cnt_domain = np.zeros(10), np.zeros(10)
        self.acc_source, self.acc_target = 0, 0
        self.cnt_source, self.cnt_target = 0, 0
        self.hit_source, self.hit_target = 0, 0

    def acc_update_mnist(self):
        Y = to_np(self.y)
        G = to_np(self.g)
        T = to_np(self.domain) # self.domain : 所属的域[0,5]
        # print('Y ',Y)
        # print('G ',G.shape)
        # print('T ',T.shape)

        # print('G ',G)
        # print('T ',T)
        T = (T * 10).astype(np.int32) #[0,9]
        # T[T >= 8] = 7 #  大于8，强制转7
        hit = (Y == G).astype(np.float32) # 是否预测正确
        # print('hit ',hit)
        is_s = to_np(self.is_source)
        # print('is_s ',is_s)
        for i in range(9): # 分别计算每个域的数量和命中的数量
            self.hit_domain[i+1] += hit[T == (i+1)].sum()  # 第i+1个域 命中的个数之和
            self.cnt_domain[i+1] += (T == (i+1)).sum()     # 该batch中，第i+1个域的样本总数
        self.acc_domain = self.hit_domain / (self.cnt_domain + 1e-10) # 每个域的命中率
        #self.acc_source, self.acc_target = self.acc_domain[5], np.concatenate([self.acc_domain[0:5],self.acc_domain[6:]],axis=0).mean()  # 合并源域和目标域
        self.acc_source, self.acc_target = self.acc_domain[1], self.acc_domain[2:10].mean()  # 合并源域和目标域
        self.acc_domain = np.round(self.acc_domain, decimals=3)
        self.acc_source = np.round(self.acc_source, decimals=3)  # 源域准确率
        self.acc_target = np.round(self.acc_target, decimals=3)  # 目标域准确率

        self.cnt_source += (is_s == 1).sum()  # 源域的数量
        self.cnt_target += (is_s == 0).sum()  # 目标域的数量

        self.hit_source += (hit[is_s == 1]).sum() #源域命中数量
        self.hit_target += (hit[is_s == 0]).sum() # 目标域命中数量
        # [1 1 1 1 4 1 2 1 1 1 1 6 1 1 1 1 3 3 1 3 1 5 1 1 1 1 5 1 1 1 1 1]
    def set_input(self, input):
        # print('set_input ')
        self.x, self.y, self.u, self.domain = input
        # print('self.u ',self.u)
        # print('self.domain ',self.domain)
        self.domain = self.domain[:, 0]/10.0
        self.u = self.u[:,0]/10.0
        
        #self.u = 
        #print('self.domain_new ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float) # asc中只要判断是否为1
        
    def print_log(self):
        print(self.loss_msg)
        print(self.acc_msg)
        with open(self.train_log, 'a') as f:
            f.write(self.loss_msg + '\n')
            f.write(self.acc_msg + '\n')
        print('acc_a ',self.acc_domain[1])
        print('acc_1 ',self.acc_domain[2])
        print('acc_2 ',self.acc_domain[3])
        print('acc_3 ',self.acc_domain[4])
        print('acc_4 ',self.acc_domain[5])
        print('acc_5 ',self.acc_domain[6])
        print('acc_6 ',self.acc_domain[7])
        print('acc_7 ',self.acc_domain[8])
        print('acc_8 ',self.acc_domain[9])
        #print('acc_9 ',self.acc_domain[9])
    def print_log2(self):
        print(self.loss_msg)
        print(self.acc_msg)
        with open(self.train_log, 'a') as f:
            f.write(self.loss_msg + '\n')
            f.write(self.acc_msg + '\n')
        print('acc_a ',self.acc_domain[5])
        print('acc_b ',self.acc_domain[4])
        print('acc_c ',self.acc_domain[6])
        print('acc_s1 ',self.acc_domain[3])
        print('acc_s2 ',self.acc_domain[7])
        print('acc_s3 ',self.acc_domain[2])
        print('acc_s4 ',self.acc_domain[8])
        print('acc_s5 ',self.acc_domain[1])
        print('acc_s6 ',self.acc_domain[9])
    def learn(self, epoch, dataloader):
        self.epoch = epoch
        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        self.acc_reset_mnist()
        # print('dataloader ',len(dataloader))
        # print('dataloader[0]',dataloader[0])
        bar = ProgressBar()
        
        for data in bar(dataloader):
            x, y, t, is_source= [to_tensor(_, self.opt.device) for _ in data]
            # print('t ',t)
            # print('is_source ',is_source)
            self.set_input(input=(x, y, t, is_source))
            self.optimize_parameters()

            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())

            self.acc_update_mnist()

        self.loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))
        self.acc_msg = '[Train][{}] Acc: source {:.3f} ({}/{}) target {:.3f} ({}/{})'.format(
            epoch, self.acc_source, self.hit_source, self.cnt_source,
            self.acc_target, self.hit_target, self.cnt_target)
        self.print_log()

        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        self.logger.add_scalar('AccSrc', self.acc_source, epoch)
        self.logger.add_scalar('AccTgt', self.acc_target, epoch)

    def test(self, epoch, dataloader, flag_save=True):
        self.eval()
        self.acc_reset()
        for data in dataloader:
            x, y, t, is_source= [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()
            self.acc_update()

        self.best_acc_tgt = max(self.best_acc_tgt, self.acc_target)
        if self.best_acc_tgt == self.acc_target and flag_save:
            self.save()

        self.acc_msg = '[Test][{}] Acc: source {:.3f} target {:.3f} best {:.3f}'.format(
            epoch, self.acc_source, self.acc_target, self.best_acc_tgt)
        self.loss_msg = ''
        self.print_log()

    def eval_mnist(self, dataloader):
        self.eval()
        self.acc_reset_mnist()
        for data in dataloader:
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()
            self.acc_update_mnist()
        self.loss_msg = ''
        self.acc_msg = f'Eval MNIST: {self.acc_domain} src: {self.acc_source} tgt:{self.acc_target}'
        self.print_log()
        return self.acc_target

    def gen_result_table(self, dataloader):

        res = PrettyTable()

        res.field_names = ["Accuracy"] + ["Source"] + [f"Target #{i}" for i in range(1, 9)]

        hit = np.zeros((10, 9))
        cnt = np.zeros((10, 9))

        for data in dataloader:
            x, y, t, is_source ,domain_label= [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source,domain_label))
            self.forward()

            Y = to_np(self.y)
            G = to_np(self.g)
            T = to_np(self.domain)
            T = (T * 10).astype(np.int32)
            T[T >= 9] = 9

            for label, pred, domain in zip(Y, G, T):
                #print(label,domain,pred)
                hit[int(label), int(domain-1)] += int(label == pred)
                cnt[int(label), int(domain-1)] += 1

        for c in range(10):
            res.add_row([f"Class {c}"] + list(np.round(100 * hit[c] / cnt[c], decimals=1)))

        res.add_row([f"Total"] + list(np.round(100 * hit.sum(0) / cnt.sum(0), decimals=1)))
        #print('average ',)
        print(res)


class DANN(BaseModel):
    def __init__(self,opt):
        super(DANN,self).__init__(opt)
        self.opt = opt
        self.netE = EncoderCNNWithoutIndex(10,activation='logsoftmax')
        # self.init_weight(self.netE)
        self.train_log = opt.outf + '/DANN2019_onehot.log'
        self.model_path = opt.outf + '/DANN2019_onehot.pth'
        # self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.netD = DiscConv(nin=opt.nz,nout=1)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['D', 'E_gan', 'E_pred']
    def set_input(self, input):
        self.x, self.y, self.u, self.domain = input
        self.u = (self.u[:,0]<=1).to(torch.float)  # 转为0 or 1
        #print('self.u ',self.u)
        self.domain = self.domain[:,0]/10.0
        # print('u ',self.u)
        # print('domain label ',self.domain_label)
        # print('self.x ',self.x.shape)
        # print('self.y ',self.y.shape)
        # print('self.u ',self.u.shape)
        # print('self.domain ',self.domain.shape)
        # self.domain = self.domain[:, 0]
        # print('self.domain_new ',self.domain)
        # print('self.domain ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float)
        self.is_target = torch.FloatTensor(self.domain.shape[0],)
        for n,u in enumerate(self.domain):
            if u*10>1 and u*10<7:
                self.is_target[n] = 1.0
            else:
                self.is_target[n] = 0
        # self.is_train_target = torch.FloatTensor(self.domain.shape[0],)
        # for n,u in enumerate(self.domain):
        #     if u*10 >1 and u*10 < 7:
        #         self.is_train_target[n] = 1.0
        #     else:
        #         self.is_train_target[n] = 0
        # print('self.domain ',self.domain)
        # print('self.is_train_target ',self.is_train_target)
        # print('self.is_source ',self.is_source)   
    def forward(self):
        self.f , self.e = self.netE(self.x, self.domain) # 换domain
        # print('self.u ',self.u.shape)
        # print('self.e ',self.e.shape)
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        self.d = self.netD(self.e)
        self.d = torch.squeeze(self.d)
        # E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1].float())
        # if len(self.d[self.is_source == 1])==0:
        #     self.loss_E_gan = -E_gan_tgt/2.0
        if len(self.d[self.is_source == 1])==0:
            self.loss_E_gan = -E_gan_tgt/2.0
        elif len(self.d[self.is_target == 1])==0:
            self.loss_E_gan = -E_gan_src/2.0
        else:
            self.loss_E_gan = -(E_gan_src + E_gan_tgt) / 2.0
        # print('self.loss_E_gan ',self.loss_E_gan)
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        # print('self.f_source ',self.f_source)
        # print('self.y_source ',self.y_source)
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        # print('self.loss_E_pred ',self.loss_E_pred)
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        # print('self.loss_E',self.loss_E)
        # self.loss_E = torch.tensor(self.loss_E, dtype=float,requires_grad=True)
        self.loss_E.backward() 

    def backward_D(self):
        self.d = self.netD(self.e.detach())
        self.d = torch.squeeze(self.d)
        # print('self.d ',self.d.shape)
        # D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # D_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        D_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1].float())
        #self.loss_D = F.mse_loss(self.d,self.u)
        # print('D_tgt ',D_tgt.dtype)
        if len(self.d[self.is_source == 1])==0:
            self.loss_D = D_tgt/2.0
        elif len(self.d[self.is_target == 1])==0:
            self.loss_D = D_src/2.0
        else:
            self.loss_D = (D_src.float() + D_tgt.float()) / 2.0
        # print('D_tgt ',D_tgt)
        # print('self.loss_D ',self.loss_D)
        # self.loss_D = torch.tensor(self.loss_D, dtype=float,requires_grad=True)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class DANN2019(BaseModel):
    def __init__(self,opt):
        super(DANN2019,self).__init__(opt)
        self.opt = opt
        self.netE = Encoder2019(10,activation='logsoftmax')
        # self.init_weight(self.netE)
        self.train_log = opt.outf + '/DANN2019_onehot.log'
        self.model_path = opt.outf + '/DANN2019_onehot.pth'
        # self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.netD = DiscConv2019(nin=opt.nz,nout=1)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['D', 'E_gan', 'E_pred']
    def set_input(self, input):
        self.x, self.y, self.u, self.domain,self.domain_label = input
        self.u = (self.u[:,0]<=1).to(torch.float)  # 转为0 or 1
        #print('self.u ',self.u)
        self.domain = self.domain[:,0]/10.0
        # print('u ',self.u)
        # print('domain label ',self.domain_label)
        # print('self.x ',self.x.shape)
        # print('self.y ',self.y.shape)
        # print('self.u ',self.u.shape)
        # print('self.domain ',self.domain.shape)
        # self.domain = self.domain[:, 0]
        # print('self.domain_new ',self.domain)
        # print('self.domain ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float)
        self.is_target = torch.FloatTensor(self.domain.shape[0],)
        for n,u in enumerate(self.domain):
            if u*10>1 and u*10<4:
                self.is_target[n] = 1.0
            else:
                self.is_target[n] = 0
        # self.is_train_target = torch.FloatTensor(self.domain.shape[0],)
        # for n,u in enumerate(self.domain):
        #     if u*10 >1 and u*10 < 7:
        #         self.is_train_target[n] = 1.0
        #     else:
        #         self.is_train_target[n] = 0
        # print('self.domain ',self.domain)
        # print('self.is_train_target ',self.is_train_target)
        # print('self.is_source ',self.is_source)   
    def forward(self):
        self.f , self.e = self.netE(self.x, self.domain) # 换domain
        # print('self.u ',self.u.shape)
        # print('self.e ',self.e.shape)
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        self.d = self.netD(self.e)
        self.d = torch.squeeze(self.d)
        # E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.domain_label[self.is_source == 1].float())
        E_gan_tgt = F.mse_loss(self.d[self.is_source == 0], self.domain_label[self.is_source == 0].float())
        # if len(self.d[self.is_source == 1])==0:
        #     self.loss_E_gan = -E_gan_tgt/2.0
        if len(self.d[self.is_source == 1])==0:
            self.loss_E_gan = -E_gan_tgt/2.0
        elif len(self.d[self.is_source == 0])==0:
            self.loss_E_gan = -E_gan_src/2.0
        else:
            self.loss_E_gan = -(E_gan_src + E_gan_tgt) / 2.0
        # print('self.loss_E_gan ',self.loss_E_gan)
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        # print('self.f_source ',self.f_source)
        # print('self.y_source ',self.y_source)
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        # print('self.loss_E_pred ',self.loss_E_pred)
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        # print('self.loss_E',self.loss_E)
        # self.loss_E = torch.tensor(self.loss_E, dtype=float,requires_grad=True)
        self.loss_E.backward() 

    def backward_D(self):
        self.d = self.netD(self.e.detach())
        self.d = torch.squeeze(self.d)
        # print('self.d ',self.d.shape)
        # D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # D_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        D_src = F.mse_loss(self.d[self.is_source == 1], self.domain_label[self.is_source == 1].float())
        D_tgt = F.mse_loss(self.d[self.is_source == 0], self.domain_label[self.is_source == 0].float())
        #self.loss_D = F.mse_loss(self.d,self.u)
        # print('D_tgt ',D_tgt.dtype)
        if len(self.d[self.is_source == 1])==0:
            self.loss_D = D_tgt/2.0
        elif len(self.d[self.is_source == 0])==0:
            self.loss_D = D_src/2.0
        else:
            self.loss_D = (D_src.float() + D_tgt.float()) / 2.0
        # print('D_tgt ',D_tgt)
        # print('self.loss_D ',self.loss_D)
        # self.loss_D = torch.tensor(self.loss_D, dtype=float,requires_grad=True)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class DANN2019_R(BaseModel):
    def __init__(self,opt):
        super(DANN2019_R,self).__init__(opt)
        self.opt = opt
        self.netE = Encoder2019(10,activation='logsoftmax')
        # self.init_weight(self.netE)
        self.train_log = opt.outf + '/DANN2019R_R.log'
        self.model_path = opt.outf + '/DANN2019R_R.pth'
        # self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.netD = DiscConv(nin=opt.nz,nout=1)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['D', 'E_gan', 'E_pred']
    def set_input(self, input):
        self.x, self.y, self.u, self.domain,self.domain_label = input
        self.u = (self.u[:,0]<=1).to(torch.float)  # 转为0 or 1
        #print('self.u ',self.u)
        self.domain = self.domain[:,0]/10.0
        # print('u ',self.u)
        # print('domain label ',self.domain_label)
        # print('self.x ',self.x.shape)
        # print('self.y ',self.y.shape)
        # print('self.u ',self.u.shape)
        # print('self.domain ',self.domain.shape)
        # self.domain = self.domain[:, 0]
        # print('self.domain_new ',self.domain)
        # print('self.domain ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float)
        self.is_target = torch.FloatTensor(self.domain.shape[0],)
        for n,u in enumerate(self.domain):
            if u*10>1 and u*10<4:
                self.is_target[n] = 1.0
            else:
                self.is_target[n] = 0
        # self.is_train_target = torch.FloatTensor(self.domain.shape[0],)
        # for n,u in enumerate(self.domain):
        #     if u*10 >1 and u*10 < 7:
        #         self.is_train_target[n] = 1.0
        #     else:
        #         self.is_train_target[n] = 0
        # print('self.domain ',self.domain)
        # print('self.is_train_target ',self.is_train_target)
        # print('self.is_source ',self.is_source)   
    def forward(self):
        self.f , self.e = self.netE(self.x, self.domain) # 换domain
        # print('self.u ',self.u.shape)
        # print('self.e ',self.e.shape)
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        self.d = self.netD(self.e)
        self.d = torch.squeeze(self.d)
        # E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1].float())
        # if len(self.d[self.is_source == 1])==0:
        #     self.loss_E_gan = -E_gan_tgt/2.0
        if len(self.d[self.is_source == 1])==0:
            self.loss_E_gan = -E_gan_tgt/2.0
        elif len(self.d[self.is_target == 1])==0:
            self.loss_E_gan = -E_gan_src/2.0
        else:
            self.loss_E_gan = -(E_gan_src + E_gan_tgt) / 2.0
        # print('self.loss_E_gan ',self.loss_E_gan)
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        # print('self.f_source ',self.f_source)
        # print('self.y_source ',self.y_source)
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        # print('self.loss_E_pred ',self.loss_E_pred)
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        # print('self.loss_E',self.loss_E)
        # self.loss_E = torch.tensor(self.loss_E, dtype=float,requires_grad=True)
        self.loss_E.backward() 

    def backward_D(self):
        self.d = self.netD(self.e.detach())
        self.d = torch.squeeze(self.d)
        # print('self.d ',self.d.shape)
        # D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # D_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        D_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1].float())
        #self.loss_D = F.mse_loss(self.d,self.u)
        # print('D_tgt ',D_tgt.dtype)
        if len(self.d[self.is_source == 1])==0:
            self.loss_D = D_tgt/2.0
        elif len(self.d[self.is_target == 1])==0:
            self.loss_D = D_src/2.0
        else:
            self.loss_D = (D_src.float() + D_tgt.float()) / 2.0
        # print('D_tgt ',D_tgt)
        # print('self.loss_D ',self.loss_D)
        # self.loss_D = torch.tensor(self.loss_D, dtype=float,requires_grad=True)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class DANN_R_RES(BaseModel):
    def __init__(self,opt):
        super(DANN_R_RES,self).__init__(opt)
        self.opt = opt
        self.netE = ResNet18()
        # self.init_weight(self.netE)
        self.train_log = opt.outf + '/DANNR_res.log'
        self.model_path = opt.outf + '/DANNR_res.pth'
        # self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.netD = DiscConv(nin=opt.nz,nout=1)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]

        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['D', 'E_gan', 'E_pred']
    def set_input(self, input):
        self.x, self.y, self.u, self.domain,self.domain_label = input
        self.u = (self.u[:,0]<=1).to(torch.float)  # 转为0 or 1
        #print('self.u ',self.u)
        self.domain = self.domain[:,0]/10.0
        # print('u ',self.u)
        # print('domain label ',self.domain_label)
        # print('self.x ',self.x.shape)
        # print('self.y ',self.y.shape)
        # print('self.u ',self.u.shape)
        # print('self.domain ',self.domain.shape)
        # self.domain = self.domain[:, 0]
        # print('self.domain_new ',self.domain)
        # print('self.domain ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float)
        self.is_target = torch.FloatTensor(self.domain.shape[0],)
        for n,u in enumerate(self.domain):
            if u*10>1 and u*10<7:
                self.is_target[n] = 1.0
            else:
                self.is_target[n] = 0
        # self.is_train_target = torch.FloatTensor(self.domain.shape[0],)
        # for n,u in enumerate(self.domain):
        #     if u*10 >1 and u*10 < 7:
        #         self.is_train_target[n] = 1.0
        #     else:
        #         self.is_train_target[n] = 0
        # print('self.domain ',self.domain)
        # print('self.is_train_target ',self.is_train_target)
        # print('self.is_source ',self.is_source)   
    def forward(self):
        self.f , self.e = self.netE(self.x, self.domain) # 换domain
        # print('self.u ',self.u.shape)
        # print('self.e ',self.e.shape)
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        self.d = self.netD(self.e)
        self.d = torch.squeeze(self.d)
        # E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        E_gan_tgt = F.mse_loss(self.d[self.is_source == 0], self.u[self.is_source == 0].float())
        # if len(self.d[self.is_source == 1])==0:
        #     self.loss_E_gan = -E_gan_tgt/2.0
        if len(self.d[self.is_source == 1])==0:
            self.loss_E_gan = -E_gan_tgt/2.0
        elif len(self.d[self.is_source == 0])==0:
            self.loss_E_gan = -E_gan_src/2.0
        else:
            self.loss_E_gan = -(E_gan_src + E_gan_tgt) / 2.0
        # print('self.loss_E_gan ',self.loss_E_gan)
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        # print('self.f_source ',self.f_source)
        # print('self.y_source ',self.y_source)
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        # print('self.loss_E_pred ',self.loss_E_pred)
        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        # print('self.loss_E',self.loss_E)
        # self.loss_E = torch.tensor(self.loss_E, dtype=float,requires_grad=True)
        self.loss_E.backward() 

    def backward_D(self):
        self.d = self.netD(self.e.detach())
        self.d = torch.squeeze(self.d)
        # print('self.d ',self.d.shape)
        # D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # D_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        D_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        D_tgt = F.mse_loss(self.d[self.is_source == 0], self.u[self.is_source == 0].float())
        #self.loss_D = F.mse_loss(self.d,self.u)
        # print('D_tgt ',D_tgt.dtype)
        if len(self.d[self.is_source == 1])==0:
            self.loss_D = D_tgt/2.0
        elif len(self.d[self.is_source == 0])==0:
            self.loss_D = D_src/2.0
        else:
            self.loss_D = (D_src.float() + D_tgt.float()) / 2.0
        # print('D_tgt ',D_tgt)
        # print('self.loss_D ',self.loss_D)
        # self.loss_D = torch.tensor(self.loss_D, dtype=float,requires_grad=True)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
class DANN2019NO(BaseModel):
    def __init__(self,opt):
        super(DANN2019NO,self).__init__(opt)
        self.opt = opt
        self.netE = Encoder2019(10,activation='logsoftmax')
        # self.init_weight(self.netE)
        self.train_log = opt.outf + '/DANN2019R_onehot.log'
        self.model_path = opt.outf + '/DANN2019R_onehot.pth'
        # self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.netD = DiscConv(nin=opt.nz,nout=1)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        #self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        #self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G]

        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['E_pred']
    def set_input(self, input):
        self.x, self.y, self.u, self.domain,self.domain_label = input
        self.u = (self.u[:,0]<=1).to(torch.float)  # 转为0 or 1
        #print('self.u ',self.u)
        self.domain = self.domain[:,0]/10.0
        # print('u ',self.u)
        # print('domain label ',self.domain_label)
        # print('self.x ',self.x.shape)
        # print('self.y ',self.y.shape)
        # print('self.u ',self.u.shape)
        # print('self.domain ',self.domain.shape)
        # self.domain = self.domain[:, 0]
        # print('self.domain_new ',self.domain)
        # print('self.domain ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float)
        self.is_target = torch.FloatTensor(self.domain.shape[0],)
        for n,u in enumerate(self.domain):
            if u*10>1 and u*10<4:
                self.is_target[n] = 1.0
            else:
                self.is_target[n] = 0
        # self.is_train_target = torch.FloatTensor(self.domain.shape[0],)
        # for n,u in enumerate(self.domain):
        #     if u*10 >1 and u*10 < 7:
        #         self.is_train_target[n] = 1.0
        #     else:
        #         self.is_train_target[n] = 0
        # print('self.domain ',self.domain)
        # print('self.is_train_target ',self.is_train_target)
        # print('self.is_source ',self.is_source)   
    def forward(self):
        self.f , self.e = self.netE(self.x, self.domain) # 换domain
        # print('self.u ',self.u.shape)
        # print('self.e ',self.e.shape)
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        # self.d = self.netD(self.e)
        # self.d = torch.squeeze(self.d)
        # # E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1])
        # # E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1])
        # E_gan_src = F.mse_loss(self.d[self.is_source == 1], self.u[self.is_source == 1].float())
        # E_gan_tgt = F.mse_loss(self.d[self.is_target == 1], self.u[self.is_target == 1].float())
        # # if len(self.d[self.is_source == 1])==0:
        # #     self.loss_E_gan = -E_gan_tgt/2.0
        # if len(self.d[self.is_source == 1])==0:
        #     self.loss_E_gan = -E_gan_tgt/2.0
        # elif len(self.d[self.is_target == 1])==0:
        #     self.loss_E_gan = -E_gan_src/2.0
        # else:
        #     self.loss_E_gan = -(E_gan_src + E_gan_tgt) / 2.0
        # # print('self.loss_E_gan ',self.loss_E_gan)
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        # print('self.f_source ',self.f_source)
        # print('self.y_source ',self.y_source)
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        # print('self.loss_E_pred ',self.loss_E_pred)
        self.loss_E =  self.loss_E_pred
        # print('self.loss_E',self.loss_E)
        # self.loss_E = torch.tensor(self.loss_E, dtype=float,requires_grad=True)
        self.loss_E.backward() 

    
    def optimize_parameters(self):
        self.forward()
        # update D
        # self.set_requires_grad(self.netD, True)
        # self.optimizer_D.zero_grad()
        # self.backward_D()
        # self.optimizer_D.step()
        # # update G
        # self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()