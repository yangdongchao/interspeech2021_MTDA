import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
        tmp_y = u
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
        #print('x_DiscConv ',x.shape)
        return self.net(x)

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.tsne = TSNE(n_components=2)
        self.pca = PCA(n_components=2)
        # self.opt.div = opt.div
        self.train_log = opt.outf + '/'+ '_pre_train.log'
        self.model_path = opt.outf +'/'+  '_pre_train.pth'
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
        T = (T * self.opt.div).astype(np.int32) #(0,8)
        # T[T >= 8] = 7 #  大于8，强制转7
        hit = (Y == G).astype(np.float32) # 是否预测正确
        # print('hit ',hit)
        is_s = to_np(self.is_source)
        # print('is_s ',is_s)
        for i in range(9): # 分别计算每个域的数量和命中的数量
            self.hit_domain[i+1] += hit[T == i+1].sum()  # 第i+1个域 命中的个数之和
            self.cnt_domain[i+1] += (T == i+1).sum()     # 该batch中，第i+1个域的样本总数
        self.acc_domain = self.hit_domain / (self.cnt_domain + 1e-10) # 每个域的命中率
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
        self.domain = self.domain[:, 0]/(10.0)
        self.u = self.u[:,0]/(10.0)
        #print('self.domain_new ',self.domain)
        self.is_source = (self.domain*10 <= 1).to(torch.float) # asc中只要判断是否为1
        
    def print_log(self):
        print(self.loss_msg)
        print(self.acc_msg)
        with open(self.train_log, 'a') as f:
            f.write(self.loss_msg + '\n')
            f.write(self.acc_msg + '\n')
        print('acc_a ',self.acc_domain[1])
        print('acc_b ',self.acc_domain[3])
        print('acc_c ',self.acc_domain[2])
        print('acc_s1 ',self.acc_domain[4])
        print('acc_s2 ',self.acc_domain[6])
        print('acc_s3 ',self.acc_domain[5])
        print('acc_s4 ',self.acc_domain[7])
        print('acc_s5 ',self.acc_domain[8])
        print('acc_s6 ',self.acc_domain[9])
    def learn(self, epoch, dataloader):
        self.epoch = epoch
        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        self.acc_reset_mnist()
        bar = ProgressBar()
        
        for data in bar(dataloader):
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
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
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
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
            #print(data)
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()
            self.acc_update_mnist()
        print('hit ',self.hit_domain)
        print('cnt ',self.cnt_domain)
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
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()

            Y = to_np(self.y)
            G = to_np(self.g)
            T = to_np(self.u)
            T = (T * self.opt.div).astype(np.int32)
            T[T >= 9] = 9

            for label, pred, domain in zip(Y, G, T):
                #print(label,domain,pred)
                hit[int(label), int(domain)-1] += int(label == pred)
                cnt[int(label), int(domain)-1] += 1

        for c in range(10):
            res.add_row([f"Class {c}"] + list(np.round(100 * hit[c] / cnt[c], decimals=1)))

        res.add_row([f"Total"] + list(np.round(100 * hit.sum(0) / cnt.sum(0), decimals=1)))
        print(res)
class Like_pre_gan(nn.Module):
    def __init__(self):
        """Initialization of the model M.
        """
        super(Like_pre_gan, self).__init__()
        self.cnn_1 = nn.Conv2d(
            in_channels=1, out_channels=48,
            kernel_size=(11, 11), stride=(2, 3), padding=5
        )
        self.cnn_2 = nn.Conv2d(
            in_channels=48, out_channels=128,
            kernel_size=5, stride=(2, 3), padding=2
        )
        self.cnn_3 = nn.Conv2d(
            in_channels=128, out_channels=192,
            kernel_size=3, stride=1, padding=1
        )
        self.cnn_4 = nn.Conv2d(
            in_channels=192, out_channels=192,
            kernel_size=3, stride=1, padding=1
        )
        self.cnn_5 = nn.Conv2d(
            in_channels=192, out_channels=128,
            kernel_size=3, stride=1, padding=0
        )
        self.max_pool_1 = nn.MaxPool2d(
            kernel_size=3, stride=(1, 2)
        )
        self.max_pool_2 = nn.MaxPool2d(
            kernel_size=3, stride=(2, 2)
        )
        self.max_pool_3 = nn.MaxPool2d(
            kernel_size=3, stride=(1, 2)
        )

        self.bn_1 = nn.BatchNorm2d(48)
        self.bn_2 = nn.BatchNorm2d(128)
        self.bn_3 = nn.BatchNorm2d(128)

    def forward(self, x,u):
        """The forward pass of the model.

        :param x: The input.
        :type x: torch.Tensor
        :return: The output of the model.
        :rtype: torch.Tensor
        """
        x = x[:,None,:,:]
        # print('x ',x.shape)
        x = x.view(x.shape[0],1,64,-1)
        #print('x ',x.shape)
        output = self.bn_1(self.max_pool_1(F.relu(self.cnn_1(x))))
        # print('output 0 ',output.shape)
        output = self.bn_2(self.max_pool_2(F.relu(self.cnn_2(output))))
        # print('output -1 ',output.shape)
        output = F.relu(self.cnn_3(output))
        # print('output -2 ',output.shape)
        output = F.relu(self.cnn_4(output))
        # print('output1 ',output.shape)
        output = self.bn_3(self.max_pool_3(F.relu(self.cnn_5(output))))
        # print('output ',output.shape)
        return output
class LabelClassifier(nn.Module):
    """The label classifier.
    """

    def __init__(self, nb_output_classes):
        """Initialization of the label classifier.

        :param nb_output_classes: The number of classes to classify\
                                 (i.e. amount of outputs).
        :type nb_output_classes: int
        """
        super(LabelClassifier, self).__init__()

        self.linear_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.linear_layers.append(nn.Linear(in_features=1536, out_features=256))
        self.linear_layers.append(nn.Linear(in_features=256, out_features=256))
        for _ in range(2):
            self.dropouts.append(nn.Dropout(.25))

        self.output_layer = nn.Linear(in_features=256, out_features=nb_output_classes)

    def forward(self, x):
        """The forward pass of the label classifier.

        :param x: The input.
        :type x: torch.Tensor
        :return: The prediction of the label classifier.
        :rtype: torch.Tensor
        """
        output = x.view(x.size()[0], -1)
        for i in range(len(self.linear_layers)):
            output = self.dropouts[i](F.relu(self.linear_layers[i](output)))
        op = self.output_layer(output)
        output = F.log_softmax(op, dim=-1)
        return output
class Discriminator(nn.Module):
    """The discriminator (domain classifier).
    """
    def __init__(self, nb_outputs):
        """Initialization of the discriminator.

        :param nb_outputs: The amount of outputs.
        :type nb_outputs: int
        """
        super(Discriminator, self).__init__()

        self.cnn_1 = nn.Conv2d(
            in_channels=128, out_channels=64,
            kernel_size=(3, 3), stride=1, padding=1
        )
        self.cnn_2 = nn.Conv2d(
            in_channels=64, out_channels=32,
            kernel_size=(3, 3), stride=1, padding=1
        )
        self.cnn_3 = nn.Conv2d(
            in_channels=32, out_channels=16,
            kernel_size=(3, 3), stride=1, padding=1
        )

        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(32)
        self.bn_3 = nn.BatchNorm2d(16)

        self.linear_1 = nn.Linear(in_features=192, out_features=nb_outputs)

    def forward(self, x):
        """The forward pass of the discriminator.

        :param x: The input.
        :type x: torch.Tensor
        :return: The prediction of the discriminator.
        :rtype: torch.Tensor
        """
        latent = self.bn_1(F.relu(self.cnn_1(x)))
        #print('latent1 ',latent.shape)
        latent = self.bn_2(F.relu(self.cnn_2(latent)))
        #print('latent2 ',latent.shape)
        latent = self.bn_3(F.relu(self.cnn_3(latent)))
        #print('latent3 ',latent.shape)
        output = latent.view(latent.size()[0], -1)

        return self.linear_1(output)
class GAN(BaseModel):  
    def __init__(self, opt):
        super(GAN, self).__init__(opt)
        self.opt = opt
        self.netE = EncoderCNNWithoutIndex(10,'logsoftmax')
        self.netD = DiscConv(nin=opt.nz, nout=opt.dim_domain)
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.loss_names = ['E_pred']
        self.lambda_gan = opt.lambda_gan
        self.train_log = opt.outf + '/'+ '_no_adapt_gan20_train.log'
        self.model_path = opt.outf +'/'+  '_no_adapt_gan20_train.pth'

    def forward(self):
        self.f, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.f.detach(), dim=1)

    def backward_G(self):
        self.loss_E_pred = F.nll_loss(self.f[self.is_source == 1], self.y[self.is_source == 1].long())
        self.loss_E_pred.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class Adaptation(BaseModel): # UADA model
    def __init__(self, opt):
        super(Adaptation, self).__init__(opt)
        self.opt = opt
        self.source_net = opt.netS
        self.target_net = opt.netE
        self.train_log = opt.outf + '/'+ '_20domain_adaptation_train.log'
        self.model_path = opt.outf +'/'+  '_20domain_adaptation_train.pth'
        self.optimizer_G = torch.optim.RMSprop(self.target_net.netE.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.RMSprop(self.target_net.netD.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

    def forward(self):
        self.source_f,self.source_e = self.source_net.netE(self.x,self.u)
        self.target_f,self.target_e = self.target_net.netE(self.x,self.u)
        # self.f, self.e = self.netE(self.x, self.u)
        self.g = torch.argmax(self.target_f.detach(), dim=1) # 只计算target的分类输出

    def backward_G(self):
        self.d = torch.sigmoid(self.target_net.netD(self.target_e)) # target模型的disc Mt(Xt)
        self.d_t = self.d[self.is_source == 0]# 只用目标域的数据
        self.loss_E_gan = -torch.log(self.d_t + 1e-10).mean()  # make target domain looks like source
        self.loss_E_pred = F.nll_loss(self.target_f[self.is_source == 1], self.y[self.is_source == 1].long())  # Mt(Xs)
        self.loss_E = self.loss_E_gan  + self.loss_E_pred
        #self.loss_E
        self.loss_E.backward()

    def backward_D(self):
        self.d = torch.sigmoid(self.target_net.netD(self.target_e.detach()))
        self.d_s_ = torch.sigmoid(self.target_net.netD(self.source_e.detach())) # 源
        self.d_s = self.d_s_[self.is_source==1] # 源
        self.d_t = self.d[self.is_source==0] # 目标
        self.loss_D = - torch.log(self.d_s + 1e-10).mean() \
                      - torch.log(1 - self.d_t + 1e-10).mean()
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.target_net.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.target_net.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


class Wadaptation(BaseModel): # WUADA model
    def __init__(self, opt):
        super(Wadaptation, self).__init__(opt)
        self.opt = opt
        self.source_net = opt.netS
        self.target_net = opt.netE
        self.train_log = opt.outf + '/'+ 'W_domain_adaptation_train.log'
        self.model_path = opt.outf +'/'+  'W_domain_adaptation_train.pth'
        self.optimizer_G = torch.optim.RMSprop(self.target_net.netE.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.optimizer_D = torch.optim.RMSprop(self.target_net.netD.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G, self.lr_scheduler_D]
        self.loss_names = ['D', 'E_gan', 'E_pred']
        self.lambda_gan = opt.lambda_gan

    def forward(self):
        self.source_f,self.source_e = self.source_net.netE(self.x,self.u)
        self.target_f,self.target_e = self.target_net.netE(self.x,self.u)
        self.g = torch.argmax(self.target_f.detach(), dim=1) # 只计算target的分类输出

    def backward_G(self):
        self.d = torch.sigmoid(self.target_net.netD(self.target_e)) # target模型的disc Mt(Xt)
        self.d_t = self.d[self.is_source == 0]# 只用目标域的数据
        self.loss_E_gan = self.d_t.mean()  # make target domain looks like source
        self.loss_E_pred = F.nll_loss(self.target_f[self.is_source == 1], self.y[self.is_source == 1].long())  # Mt(Xs)
        self.loss_E = self.loss_E_gan  + self.loss_E_pred
        #self.loss_E
        self.loss_E.backward()

    def backward_D(self):
        self.d = torch.sigmoid(self.target_net.netD(self.target_e.detach()))
        self.d_s_ = torch.sigmoid(self.target_net.netD(self.source_e.detach())) # 源
        self.d_s = self.d_s_[self.is_source==1]
        self.d_t = self.d[self.is_source==0]
        self.loss_D = self.d_s.mean() - self.d_t.mean() # w-gan的公式
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.target_net.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.target_net.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
