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
import math
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
from typing import Optional, Sequence
import torch
import torch.nn as nn
# from kernels import optimal_kernel_combinations
# from kernels import GaussianKernel
from typing import Optional, List
from qpsolvers import solve_qp




class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Parameters:
        - sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        - track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        - alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


def optimal_kernel_combinations(kernel_values: List[torch.Tensor]) -> torch.Tensor:
    # use quadratic program to get optimal kernel
    num_kernel = len(kernel_values)
    kernel_values_numpy = array([float(k.detach().cpu().data.item()) for k in kernel_values])
    if np.all(kernel_values_numpy <= 0):
        beta = solve_qp(
            P=-np.eye(num_kernel),
            q=np.zeros(num_kernel),
            A=kernel_values_numpy,
            b=np.array([-1.]),
            G=-np.eye(num_kernel),
            h=np.zeros(num_kernel),
        )
    else:
        beta = solve_qp(
            P=np.eye(num_kernel),
            q=np.zeros(num_kernel),
            A=kernel_values_numpy,
            b=np.array([1.]),
            G=-np.eye(num_kernel),
            h=np.zeros(num_kernel),
        )
    beta = beta / beta.sum(axis=0) * num_kernel  # normalize
    return sum([k * b for (k, b) in zip(kernel_values, beta)])


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Multiple Kernel Maximum Mean Discrepancy (MK-MMD) used in
    `Learning Transferable Features with Deep Adaptation Networks <https://arxiv.org/pdf/1502.02791>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations as :math:`\{z_i^s\}_{i=1}^{n_s}` and :math:`\{z_i^t\}_{i=1}^{n_t}`.
    The MK-MMD :math:`D_k (P, Q)` between probability distributions P and Q is defined as

    .. math::
        D_k(P, Q) \triangleq \| E_p [\phi(z^s)] - E_q [\phi(z^t)] \|^2_{\mathcal{H}_k},

    :math:`k` is a kernel function in the function space

    .. math::
        \mathcal{K} \triangleq \{ k=\sum_{u=1}^{m}\beta_{u} k_{u} \}

    where :math:`k_{u}` is a single kernel.

    Using kernel trick, MK-MMD can be computed as

    .. math::
        \hat{D}_k(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} k(z_i^{s}, z_j^{s}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} k(z_i^{t}, z_j^{t}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} k(z_i^{s}, z_j^{t}). \\

    Parameters:
        - **kernels** (tuple(`nn.Module`)): kernel functions.
        - **linear** (bool): whether use the linear version of DAN. Default: False
        - **quadratic_program** (bool): whether use quadratic program to solve :math:`\beta`. Default: False

    Inputs: z_s, z_t
        - **z_s** (tensor): activations from the source domain, :math:`z^s`
        - **z_t** (tensor): activations from the target domain, :math:`z^t`

    Shape:
        - Inputs: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels.

    Examples::
        >>> from dalib.modules.kernels import GaussianKernel
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
        >>> loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
        >>> # features from source domain and target domain
        >>> z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss(z_s, z_t)
    """

    def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False,
                 quadratic_program: Optional[bool] = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        self.quadratic_program = quadratic_program

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        if not self.quadratic_program:
            kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
            # Add 2 / (n-1) to make up for the value on the diagonal
            # to ensure loss is positive in the non-linear version
            loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
        else:
            kernel_values = [(kernel(features) * self.index_matrix).sum() + 2. / float(batch_size - 1) for kernel in self.kernels]
            loss = optimal_kernel_combinations(kernel_values)
        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
                         linear: Optional[bool] = True) -> torch.Tensor:
    r"""
    Update the `index_matrix` which convert `kernel_matrix` to loss.
    If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
    Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
    """
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
    return index_matrix

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


class EncoderMMDWithoutIndex(nn.Module):
    def __init__(self, classes_num, activation):
        super(EncoderMMDWithoutIndex, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        #self.feature = None
        self.target = None
        # self.z_adapt_feature = None
        # self.adatp_fc = nn.Linear(512,512)
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
        # z_adapt = self.adatp_fc(z)
        # print('z_max ',z.shape)
        E_z = z
        # z_o = z_adapt
        z = self.fc(z)
        # print('z_fc ',z.shape)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(z, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(z)
        # print('output ',output.shape)
        #self.feature = E.detach()
        self.target = tmp_y.detach()
        # self.z_adapt_feature = z_o.detach() # 保存中间变量
        return output,E_z
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

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.tsne = TSNE(n_components=2)
        self.pca = PCA(n_components=2)

        self.train_log = opt.outf + '/MMD.log'
        self.model_path = opt.outf + '/MMD.pth'
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
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
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
            x, y, t, is_source = [to_tensor(_, self.opt.device) for _ in data]
            self.set_input(input=(x, y, t, is_source))
            self.forward()

            Y = to_np(self.y)
            G = to_np(self.g)
            T = to_np(self.domain)
            #print('self.u ',self.u)
            T = (T * 10).astype(np.int32)
            #print('T ',T)
            T[T >= 9] = 9

            for label, pred, domain in zip(Y, G, T):
                #print(label,domain,pred)
                hit[int(label), int(domain-1)] += int(label == pred)
                cnt[int(label), int(domain-1)] += 1

        for c in range(10):
            #print('hit,cnt ',hit[c],cnt[c])
            res.add_row([f"Class {c}"] + list(np.round(100 * hit[c] / cnt[c], decimals=1)))

        res.add_row([f"Total"] + list(np.round(100 * hit.sum(0) / cnt.sum(0), decimals=1)))
        #print('average ',)
        print(res)

class Classifier(nn.Module):
    def __init__(self, in_dim,out_dim):
        super(Classifier, self).__init__()
        self.fc_adapt = nn.Linear(in_dim,in_dim)
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.feature = None
        #self.target = None
        self.init_weights()
    def init_weights(self):
        init_layer(self.fc)
        init_layer(self.fc_adapt)
    def forward(self, x):

        """
        :param x: (batch_size, bins)
        :return:
        """
        x = self.fc_adapt(x)
        A_x = x
        z = self.fc(x)
        # print('z_fc ',z.shape)
        output = F.log_softmax(z, dim=-1)
        self.feature = A_x.detach()
        return output, x

class MMD(BaseModel):
    def __init__(self,opt):
        super(MMD,self).__init__(opt)
        self.opt = opt
        self.netE = EncoderMMDWithoutIndex(10,activation='logsoftmax')
        self.optimizer_G = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.train_log = opt.outf + '/MMD_pre_train.log'
        self.model_path = opt.outf + '/MMD_pre_train.pth'
        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['E_pred']
    def set_input(self, input):
        self.x, self.y, self.u, self.domain = input
        self.u = (self.u[:,0]<=1).to(torch.float)
        #print('self.u ',self.u)
        self.domain = self.domain[:,0]/10.0
        self.is_source = (self.domain*10 <= 1).to(torch.float)
    def forward(self):
        self.f , self.e= self.netE(self.x, self.domain) # 换domain
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        self.loss_E_pred.backward() 

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class Adapt_MMD(BaseModel):
    def __init__(self,opt):
        super(Adapt_MMD,self).__init__(opt)
        self.opt = opt
        self.source_net = opt.netS
        # self.target_net = opt.netT
        self.classifier = Classifier(512,10)
        self.optimizer_G = torch.optim.Adam([
            {'params': self.source_net.netE.conv_block3.parameters()},
            {'params': self.source_net.netE.conv_block4.parameters()},
            {'params': self.classifier.parameters(),'lr': opt.lr*10}], lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        #self.optimizer_G = torch.optim.Adam(self.source_net.netE.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=not False, quadratic_program=False)
        # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 50))
        # self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1 / 50))
        self.lr_schedulers = [self.lr_scheduler_G]
        self.train_log = opt.outf + '/MMD_adapt_train.log'
        self.model_path = opt.outf + '/MMD_adapt_train.pth'
        self.lambda_gan = opt.lambda_gan
        self.loss_names = ['E_pred']

    def forward(self):
        self.f_tmp ,self.e = self.source_net.netE(self.x, self.domain) # 换domain
        # print('self.e ',self.e.shape)
        self.f, self.adapt = self.classifier(self.e)
        # print('self.adapt ',self.adapt.shape)
        self.g = torch.argmax(self.f.detach(), dim=1)  # self.g为最后的预测
    def backward_G(self):
        self.y_source = self.y[self.is_source == 1]
        self.f_source = self.f[self.is_source == 1]
        self.adapt_s = self.adapt[self.is_source==1] # 源域的adapt feature
        self.adapt_t = self.adapt[self.is_source==0] # 目标域的 adapt feature
        # print('self.f_source ',self.f_source)
        # print('self.y_source ',self.y_source)
        self.loss_E_pred = F.nll_loss(self.f_source, self.y_source.long())
        if self.adapt_s.shape[0]<=1 or self.adapt_t.shape[0]<=1:
            self.loss = self.loss_E_pred
        else:
            # print(self.adapt_s.shape,self.adapt_t.shape)
            mini_batch = min(self.adapt_s.shape[0],self.adapt_t.shape[0])
            #print('mini_batch ',mini_batch)
            self.transfer_loss = self.mkmmd_loss(self.adapt_s[0:mini_batch,:],self.adapt_t[0:mini_batch,:])
            #print('self.transfer_loss ',self.transfer_loss)
            self.loss = self.transfer_loss*self.lambda_gan + self.loss_E_pred
        self.loss.backward()
        

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
