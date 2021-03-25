from easydict import EasyDict
import sys
sys.path.append('/home/ydc/code2/CIDA-ASC')
#print(sys.path)
from models.GAN_ASC import Adaptation,GAN
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import ASC2020,MyDataset,ASC2020_pre
def print_args(opt):
    for k, v in opt.items():
        print(f'{k}: {v}')


opt = EasyDict()
# choose the method from ["CIDA", "PCIDA", "SO", "ADDA", "DANN" "CUA"]
opt.model = "gan"
# choose run on which device ["cuda", "cpu"]
opt.device = "cuda"

if opt in ['CIDA', 'PCIDA']:
        opt.dropout = 0.2
else:
    opt.dropout = 0

# model input/hidden/output size
opt.nh = 512  # size of hidden neurons
opt.nc = 10  # number of class
opt.nz = 256  # size of features

# training parameteres
opt.num_epoch = 100
opt.batch_size = 32
opt.lr = 2e-4
opt.weight_decay = 5e-4
opt.beta1 = 0.9

# loss parameters
opt.lambda_gan = 1

# experiment folder
opt.shebei = 'a'
opt.device_to_num = 1
opt.div = 10
opt.exp = 'gan18_last_version' + opt.model
opt.outf = './dump/' + opt.exp
os.system('mkdir -p ' + opt.outf)
print('Traing result will be saved in ', opt.outf)

# dataset info
opt.dim_domain = 1

# parameters for CUA
opt.continual_da = (opt.model == 'CUA')
if opt.model == 'CUA':
    opt.num_da_step = 7
    opt.num_epoch_pre = 10
    opt.num_epoch_sub = 10
    opt.lr_decay_period = 50
    opt.lambda_rpy = 1.0
    
print_args(opt)

pre_train = True
# build dataset and data loader
dataset_train = ASC2020_pre('/home/ydc/code2/CIDA-ASC/data/feature_2018/2018_train_split.h5') # 使用all data
# print('dataset_train ',len(dataset_train))
train_dataloader = DataLoader(
    dataset=dataset_train,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
#print('train_dataloader ',len(train_dataloader))
dataset_test = ASC2020_pre('/home/ydc/code2/CIDA-ASC/data/feature_2018/2018_train_split.h5')
test_dataloader = DataLoader(
    dataset=dataset_test,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
# build model
model_pool = {
    # 'SO': SO,
    'adapt': Adaptation,
    'gan': GAN
    # 'ADDA': ADDA,
    # 'DANN': DANN,
    # 'CUA': CUA,
}
source_model = model_pool['gan'](opt)
source_model = source_model.to(opt.device)

target_model = model_pool['gan'](opt)
target_model = target_model.to(opt.device)
print(source_model)
# source_model.load_state_dict(torch.load("/home/ydc/code2/CIDA-ASC/Main/dump/gan18gan/_gan20_train.pth"))
# source_model.eval_mnist(test_dataloader)

if pre_train:
    source_model.load_state_dict(torch.load("/home/ydc/code2/CIDA-ASC/dump/gan18_last_versiongan/_no_adapt_gan20_train.pth"))
    target_model.load_state_dict(torch.load("/home/ydc/code2/CIDA-ASC/dump/gan18_last_versiongan/_no_adapt_gan20_train.pth"))
    opt.netS = source_model
    opt.netE = target_model
    model=model_pool['adapt'](opt)
    model = model.to(opt.device)
    #model.load_state_dict(torch.load('/home/ydc/code2/CIDA-ASC/Main/dump/gan20_last_versiongan/_20domain_adaptation_train.pth'))
    print(model)
    # model.load_state_dict(torch.load('/home/ydc/code2/CIDA-ASC/dump/adapt_trainadapt/_domain_adaptation_train.pth'))
    # model.eval_mnist(train_dataloader)
    # model.gen_result_table(train_dataloader)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    best_acc_target = 0
    if not opt.continual_da:
        # Single Step Domain Adaptation
        # print('len...train_dataloader  ',len(train_dataloader))
        for epoch in range(opt.num_epoch):
            print('epoch ',epoch)
            model.learn(epoch, train_dataloader)
            if (epoch + 1) % 10 == 0:
                acc_target = model.eval_mnist(test_dataloader)
                if acc_target > best_acc_target:
                    print('Best acc target. saved.')
                    model.save()
    model.gen_result_table(test_dataloader)

else:
    best_acc_target = 0
    for epoch in range(opt.num_epoch):
        print('epoch ',epoch)
        source_model.learn(epoch, train_dataloader)
        if (epoch + 1) % 10 == 0:
            acc_target = source_model.eval_mnist(test_dataloader)
            if acc_target > best_acc_target:
                print('Best acc target. saved.')
                source_model.save()
    source_model.eval_mnist(test_dataloader)

# target_model  = model_pool['ADDA'](opt)
# target_model = target_model.to(opt.device)
# target_model.load_state_dict(torch.load("/home/ydc/code2/CIDA-ASC/dump/pre_trainADDA/_pre_train.pth"))

# opt.netS = source_model
# opt.netE = target_model
# model=model_pool[opt.model](opt)
# model = model.to(opt.device)
# print(model)
# # model.load_state_dict(torch.load('/home/ydc/code2/CIDA-ASC/dump/adapt_trainadapt/_domain_adaptation_train.pth'))
# # model.eval_mnist(train_dataloader)
# # model.gen_result_table(train_dataloader)
# # Find total parameters and trainable parameters
# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')


# # """
# # Training the model from the scratch
# # """
# best_acc_target = 0
# if not opt.continual_da:
#     # Single Step Domain Adaptation
#     # print('len...train_dataloader  ',len(train_dataloader))
#     for epoch in range(opt.num_epoch):
#         print('epoch ',epoch)
#         model.learn(epoch, train_dataloader)
#         if (epoch + 1) % 10 == 0:
#             acc_target = model.eval_mnist(train_dataloader)
#             if acc_target > best_acc_target:
#                 print('Best acc target. saved.')
#                 model.save()
# model.gen_result_table(train_dataloader)