from easydict import EasyDict
import sys
sys.path.append('/home/ydc/code2/CIDA-ASC')
#print(sys.path)
from models.new_index_model import CIDA, PCIDA
import os
from torch.utils.data import DataLoader
from data.dataset import ASC2020,MyDataset,domain_index
def print_args(opt):
    for k, v in opt.items():
        print(f'{k}: {v}')


opt = EasyDict()
# choose the method from ["CIDA", "PCIDA", "SO", "ADDA", "DANN" "CUA"]
opt.model = "PCIDA"
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
opt.batch_size = 48
opt.lr = 2e-4
opt.weight_decay = 5e-4
opt.beta1 = 0.9

# loss parameters
opt.lambda_gan = 2.0

# experiment folder
opt.exp = 'CIDACNN_' + opt.model
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
print(domain_index)
dataset_train = ASC2020('/home/ydc/code2/CIDA-ASC/data/feature/new_data_dict.h5')
# print('dataset_train ',len(dataset_train))
train_dataloader = DataLoader(
    dataset=dataset_train,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
# print('train_dataloader ',len(train_dataloader))
#dataset_test = ASC2020('/home/ydc/code2/CIDA-ASC/data/feature/test.h5')
test_dataloader = DataLoader(
    dataset=dataset_train,
    shuffle=True,
    batch_size=opt.batch_size,
    num_workers=4,
)
# build model
model_pool = {
    # 'SO': SO,
    'CIDA': CIDA,
    'PCIDA': PCIDA,
    # 'ADDA': ADDA,
    # 'DANN': DANN,
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
else:
    pass
    # continual DA training
    # continual_dataset = ContinousRotateMNIST()

    # print('===> pretrain the classifer')
    # model.prepare_trainer(init=True)
    # for epoch in range(opt.num_epoch_pre):
    #     model.learn(epoch, train_dataloader, init=True)
    #     if (epoch + 1) % 10 == 0:
    #         model.eval_mnist(test_dataloader)
    # print('===> start continual DA')
    # model.prepare_trainer(init=False)
    # for phase in range(opt.num_da_step):
    #     continual_dataset.set_phase(phase)
    #     print(f'Phase {phase}/{opt.num_da_step}')
    #     print(f'#source {len(continual_dataset.ds_source)} #target {len(continual_dataset.ds_target[phase])} #replay {len(continual_dataset.ds_replay)}')
    #     continual_dataloader = DataLoader(
    #         dataset=continual_dataset,
    #         shuffle=True,
    #         batch_size=opt.batch_size,
    #         num_workers=4,
    #     )
    #     for epoch in range(opt.num_epoch_sub):
    #         model.learn(epoch, continual_dataloader, init=False)
    #         if (epoch + 1) % 10 == 0:
    #             model.eval_mnist(test_dataloader)

    #     target_dataloader = DataLoader(
    #         dataset=continual_dataset.ds_target[phase],
    #         shuffle=True,
    #         batch_size=opt.batch_size,
    #         num_workers=4,
    #     )
    #     acc_target = model.eval_mnist(test_dataloader)
    #     if acc_target > best_acc_target:
    #         print('Best acc target. saved.')
    #         model.save()
    #     data_tuple = model.gen_data_tuple(target_dataloader)
    #     continual_dataset.ds_replay.update(data_tuple)  
model.gen_result_table(test_dataloader)