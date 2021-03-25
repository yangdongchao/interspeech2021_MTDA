from easydict import EasyDict
import sys
sys.path.append('/home/ydc/code2/CIDA-ASC')
#print(sys.path)
from models.MCD_model import MCD
import torch
import os
from torch.utils.data import DataLoader
from data.dataset import ASC2020,MyDataset,ASC2020_pre
def print_args(opt):
    for k, v in opt.items():
        print(f'{k}: {v}')


opt = EasyDict()
# choose the method from ["CIDA", "PCIDA", "SO", "ADDA", "DANN" "CUA"]
opt.model = "MCD"
# choose run on which device ["cuda", "cpu"]
opt.device = "cuda"

if opt in ['CIDA', 'PCIDA','MCD']:
        opt.dropout = 0.2
else:
    opt.dropout = 0

# model input/hidden/output size
opt.nh = 512  # size of hidden neurons
opt.nc = 10  # number of class
opt.nz = 256  # size of features

# training parameteres
opt.num_epoch = 50
opt.batch_size = 54
opt.lr = 2e-4
opt.weight_decay = 5e-4
opt.beta1 = 0.9

# loss parameters
opt.lambda_gan = 1
opt.num_k  = 4
# experiment folder
opt.exp = '20_split_MCD_' + opt.model
opt.outf = '/home/ydc/code2/CIDA-ASC/dump/' + opt.exp
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
# build dataset and data loader
#all_data = ASC2020('/home/ydc/code2/dcase2019_task1-master/workplace/features/logmel_64frames_64melbins/TAU-urban-acoustic-scenes-2020-mobile-development.h5')
dataset_train = ASC2020_pre('/home/ydc/code2/CIDA-ASC/data/final_feature/final_train')
# print('dataset_train ',len(dataset_train))
train_dataloader = DataLoader(
    dataset=dataset_train,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
# print('train_dataloader ',len(train_dataloader))
dataset_test = ASC2020_pre('/home/ydc/code2/CIDA-ASC/data/final_feature/final_test')
test_dataloader = DataLoader(
    dataset=dataset_test,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
# build model
model_pool = {
    # 'SO': SO,
    'MCD': MCD
    #'ADDA': ADDA,
    #'DANN': DANN,
    # 'CUA': CUA,
}
model = model_pool[opt.model](opt)
model = model.to(opt.device)
model.load_state_dict(torch.load('/home/ydc/code2/CIDA-ASC/dump/20_split_MCD_MCD/MCD.pth'))
print(model)


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')


"""
Training the model from the scratch
"""
best_acc_target = 0
if not opt.continual_da:
    for epoch in range(opt.num_epoch):
        print('epoch ',epoch)
        model.learn(epoch, train_dataloader)
        if (epoch + 1) % 10 == 0:
            acc_target = model.eval_mnist(test_dataloader)
            if acc_target > best_acc_target:
                print('Best acc target. saved.')
                model.save()
else:
    pass
model.gen_result_table(test_dataloader)
