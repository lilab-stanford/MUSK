from email.mime import base
from CA25net import *
import scipy.io
import numpy as np
import time
import torch
# from shutil import copyfile

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

base_dir = '/root/autodl-tmp/pycharm_project_CA2.5'
train_path = '/root/autodl-tmp/MoNuSeg2018/standard/train'
test_path = '/root/autodl-tmp/MoNuSeg2018/standard/test'


epoch = 50
batch_size = 1
deline_lr_epoch = 3
show_epoch = 1
weight_of_mask = 0.65
lr = 1e-4
# lr = 5e-5
gamma = 0.95

w_dir = '{}/weights'.format(base_dir)
if not os.path.exists(w_dir):
    os.makedirs(w_dir)

r_dir = '{}/results'.format(base_dir)
if not os.path.exists(r_dir):
    os.makedirs(r_dir)

print(base_dir)

# copyfile('CA25.py','{}/CA25.py'.format(base_dir))

f = open('{}/log.txt'.format(base_dir), 'w')


training_data = XuDataset(train_path)
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

test_data = XuDataset(test_path)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = Cia().to(device)

# checkpoint = torch.load('weights/ep61_loss0.7928918600082397.ptf')
# checkpoint = torch.load('abweights/CA25_initial.ptf')
# model.load_state_dict(checkpoint['model_state_dict'])


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Mask acc : 0.9201083183288574 --  Boundary acc : 0.5252535939216614
#torch.save({'model_state_dict': model.state_dict()}, 'abweights/CA25_initial.ptf')

print('Begin training.')
start = time.time()

loss_f, macc_f, bacc_f, cacc_f = [], [], [], []

for ep in range(epoch):
    w = weight_of_mask

#0.75 - np.exp(2*ep/epoch)/(2*np.exp(2))
#    if ep < 70:
#        w = 0.8
#    else:
#        w = 0.2

    for bt, data in enumerate(train_loader):
        model.train()
        img, label, bound = data
        img = img.cuda()
        label = label.cuda()
        bound = bound.cuda()

        mout, bout = model(img)
        #print(bout.shape)
        #loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, ep/(2*epoch)+1/4)
        # loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0.3) #w是控制黏连边权重
        loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0) #w是控制黏连边权重，置零0后CA2.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if ep % deline_lr_epoch  == 0:
        lr = lr * gamma
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    if ep % show_epoch  == 0:  
        acc_all, bacc_all, cacc_all, loss_all = [], [], [], []
        with torch.no_grad():
            for verify in test_loader:
                img, label, bound = verify
                img = img.cuda()
                label = label.cuda()
                bound = bound.cuda()
                model.eval()
                mout, bout = model(img)
                # loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0.5)
                loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0) #w是控制黏连边权重，置零0后CA2.0
                loss_all.append(loss.cpu().numpy())
                acc = dice_acc(mout[0][0], label)
                acc_all.append(acc)
                acc = dice_acc(bout[0][0], bound>0)
                bacc_all.append(acc)
                acc = dice_acc(bout[0][1], bound>1)
                cacc_all.append(acc)

            acc_all = np.array(acc_all)
            loss_all = np.array(loss_all)
            bacc_all = np.array(bacc_all)
            cacc_all = np.array(cacc_all)
            message = 'epoch num : {} -- Loss: {} -- Mask acc : {} --  Boundary acc : {} -- Clustered edge acc : {} \n'.format(ep + 1, loss_all.mean(), acc_all.mean(), bacc_all.mean(), cacc_all.mean())
            print(message)
            f.write(message)
            f.flush()

            loss_f.append(loss_all.mean())
            macc_f.append(acc_all.mean())
            bacc_f.append(bacc_all.mean())
            cacc_f.append(cacc_all.mean())
            
            # if ep > 9 and ep % 10 == 0 :
                #lr = lr * 0.6
            if ep % (epoch-1) == 0:
                torch.save({'model_state_dict': model.state_dict()}, '{}/ep{}_loss{}.ptf'.format(w_dir,ep+1,bacc_all.mean()))

macc_f = np.array(macc_f)
loss_f = np.array(loss_f)
bacc_f = np.array(bacc_f)
cacc_f = np.array(cacc_f)
mdic = {"macc":macc_f, "loss":loss_f,"bacc":bacc_f, "cacc":cacc_f}
scipy.io.savemat("{}/cl_train.mat".format(r_dir), mdic)

torch.save({'model_state_dict': model.state_dict()}, '{}/CA25_n.ptf'.format(w_dir))
end = time.time()
print('Total training time is {}h'.format((end-start)/3600))
print('Finished Training')

f.write('Total training time is {}h\n'.format((end-start)/3600))
f.close()

# %%

# loss_all,acc_all = [],[]
# with torch.no_grad():
#     for verify in train_loader:
#         img, label = verify
#         img = img.cuda()
#         label = label.cuda()
#         model.eval()
#         out = model(img)
#         loss = my_loss(out, label)
#         acc = my_acc(out, label)
#         acc_all.append(acc)
#         loss_all.append(loss.cpu().numpy())

#     acc_all = np.array(acc_all)
#     loss_all = np.array(loss_all)
#     print('Loss : {} -- Acc : {} -- Max Acc : {} -- Min Acc : {}'.format(loss_all.mean(), acc_all.mean(), acc_all.max(), acc_all.min()))

# mdic = {"loss_mean":loss_noCL, "loss_max":loss_max,"loss_std":loss_std, "acc_test":acc_all}
# scipy.io.savemat("result/noCL_results.mat", mdic)
# torch.save({
#             'epoch': ep,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, 'Unet_noCL.ptf')