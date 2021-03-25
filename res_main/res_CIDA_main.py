from easydict import EasyDict
import sys
sys.path.append('/home/ydc/code2/CIDA-ASC')
#print(sys.path)
from models.model import CIDA, PCIDA,CIDA_RES
import os
from torch.utils.data import DataLoader
from data.dataset import ASC2020,MyDataset,ASC2020_pre
def print_args(opt):
    for k, v in opt.items():
        print(f'{k}: {v}')


opt = EasyDict()
# choose the method from ["CIDA", "PCIDA", "SO", "ADDA", "DANN" "CUA"]
opt.model = "CIDA_res"
# choose run on which device ["cuda", "cpu"]
opt.device = "cuda"

if opt in ['CIDA', 'PCIDA','CIDA_res']:
        opt.dropout = 0.2
else:
    opt.dropout = 0

# model input/hidden/output size
opt.nh = 512  # size of hidden neurons
opt.nc = 10  # number of class
opt.nz = 256  # size of features

# training parameteres
opt.num_epoch = 200
opt.batch_size = 24
opt.lr = 1e-4
opt.weight_decay = 5e-4
opt.beta1 = 0.9

# loss parameters
opt.lambda_gan = 1.5

# experiment folder
opt.exp = '20_RES_CIDA_' + opt.model
opt.outf = './' + opt.exp
os.system('mkdir -p ' + opt.outf)
print('Traing result will be saved in ', opt.outf)

# dataset info
opt.dim_domain = 1

# parameters for CUA
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
print(len(train_dataloader))
# print('train_dataloader ',len(train_dataloader))
dataset_test = ASC2020_pre('/home/ydc/code2/CIDA-ASC/data/final_feature/final_test')
test_dataloader = DataLoader(
    dataset=dataset_test,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
print(len(test_dataloader))
# build model
model_pool = {
    # 'SO': SO,
    'CIDA': CIDA,
    'PCIDA': PCIDA,
    'CIDA_res': CIDA_RES
    #'ADDA': ADDA,
    #'DANN': DANN,
    # 'CUA': CUA,
}
model = model_pool[opt.model](opt)
model = model.to(opt.device)
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
for epoch in range(opt.num_epoch):
    print('epoch ',epoch)
    model.learn(epoch, train_dataloader)
    if (epoch + 1) % 10 == 0:
        acc_target = model.eval_mnist(test_dataloader)
        if acc_target > best_acc_target:
            print('Best acc target. saved.')
            model.save()
model.gen_result_table(test_dataloader)