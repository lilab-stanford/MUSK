from CIAnet import *
import scipy.io
import numpy as np
import time
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import yaml
import random
# from Dataset import ciaData
from transform import color_norm_jitter, HEDJitter, LABJitter, LABJitter_hsv, Dynamic_P_class, ConcatDataset #2.26添加
import argparse
import logging #2.25添加
import copy #2.26添加
import json #2.25添加

_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Dataset parameters
parser.add_argument('--dataset', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output', type=str, metavar='DIR',
                    help='path to output dir')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('-b', '--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
############## cj
# 2.9加入，cj的p控制
parser.add_argument('--color-jitter', nargs='+', type=float, default=None,
                    help='Color jitter factor Brigtness-Contrast-S-Hue(default: [0, 0, 0, 0])')
# 12.26加入，控制HEDJitter
parser.add_argument('--hed-jitter', type=float, default=None,
                    help='HED-jitter factory(default: 0)')
parser.add_argument('--lab-jitter',  nargs='+', type=float, default=None,
                    help='LAB-jitter factory(default: None)')
parser.add_argument('--cj-p', type=float, default=1.0, metavar='PCT',
                    help='color jitter possibility (default: 1, range: 0-1)')

############# 12.20加入，norm&jitter参数配置 ##########
parser.add_argument('--nj-config', type=str, default=None, metavar='PCT',
                    help='norm&jitter yaml config path (default: '')')
parser.add_argument('--nj-stdhyper', type=float, default=0.0, metavar='PCT',
                    help='norm&jitter std hyper (default: 0)')
parser.add_argument('--nj-distribution', type=str, default=None, metavar='PCT',
                    help='norm&jitter distribution (default: '')')
parser.add_argument('--nj-p', type=float, default=1.0, metavar='PCT', #2.9加入，nj的p控制
                    help='norm&jitter possibility (default: 1, range: 0-1)')
############# 2.9加入，norm&jitter强度控制 ##########
parser.add_argument('--nj-dynamic',action='store_true', default=False,
                    help='Enable norm-jitter dynamic-p (default: False)')
parser.add_argument('--dynamic-factor', type=float, default=1.0,
                    help='norm-jitter dynamic-p factor(default: 1)')
############# 2.9加入，nj-TTA ##########
parser.add_argument('--nj-TTA', type=int, default=0,
                    help='Enable norm-jitter Test Time Augmentation (default: 0)')
# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

#12.20增加，获取yaml文件的参数
def get_yaml_data(yaml_file):
    # 打开yaml文件
    # _logger.info(yaml_file)
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()

    # 将字符串转化为字典或列表
    # print("***转化yaml数据为字典或列表***")
    data = yaml.load(file_data, Loader=yaml.FullLoader)

    return data
    
if __name__ == '__main__':
    args, args_text = _parse_args()
    
    # 12.16修改，加入各个随机种子
    # random_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    # epoch, batch = 50, 2
    epoch, batch = args.epochs, args.batch_size
    train_path = os.path.join(args.dataset,'train') #'/root/autodl-tmp/MoNuSeg2018/standard/train'
    val_path = os.path.join(args.dataset,'val')
    test_path = os.path.join(args.dataset,'test') #'/root/autodl-tmp/MoNuSeg2018/standard/test'
    base_dir = '/root/autodl-tmp/pycharm_project_CA2.5'
    w_dir = '{}/weights'.format(base_dir)
    o_dir = '{}/outputs/{}'.format(base_dir, args.output)
    if not os.path.exists(w_dir):
        os.makedirs(w_dir)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)
    if not os.path.exists(os.path.join(o_dir,'test')):
        os.makedirs(os.path.join(o_dir,'test'))
    if not os.path.exists(os.path.join(o_dir,'val')):
        os.makedirs(os.path.join(o_dir,'val'))
    '''
    x = torch.from_numpy(x).float()
    x = x / 255  # normalization
    x = x.unsqueeze(1)
    y = torch.from_numpy(y).to(torch.long)
    z = torch.from_numpy(z).to(torch.long)
    '''

    #1.20添加，将输出保存在一个log里
    _logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("{}/output_info.log".format(o_dir))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    
    
    # train_size = 250
    transform_list = []
    
    if args.nj_config is not None:
        nj_config = get_yaml_data(args.nj_config)
        
        nj_stdhyper = args.nj_stdhyper
        nj_distribution = args.nj_distribution #1.30添加，手工指定6个采样的分布
        nj_p = args.nj_p #2.9添加，调整nj的概率
        
        nj_config['std_hyper'] = nj_stdhyper
        nj_config['distribution'] = nj_distribution
        nj_config['p'] = nj_p
        norm_jitter = nj_config
        
        # LAB / HED
        # 根据config文件的空间来决定是哪种
        # 12.26好像有问题，之前不知道顺序，很可怕
        if norm_jitter['methods'] == 'Reinhard':
            # 1.10修改，nj方法的lab和hsv进行统一
            # 1.30修改，lab，hsv和hed方法统一，根据color_space确定
            if norm_jitter['color_space'] == 'LAB' or norm_jitter['color_space'] == 'HSV' or norm_jitter['color_space'] == 'HED':
                color_space = norm_jitter['color_space'] #获取颜色空间名称
                
                #1.30修改，avg和std已经自带分布，在transform里面修改即可
                mean_dataset = [norm_jitter[color_space[0]]['avg'],norm_jitter[color_space[1]]['avg'],norm_jitter[color_space[2]]['avg']]
                std_dataset = [norm_jitter[color_space[0]]['std'],norm_jitter[color_space[1]]['std'],norm_jitter[color_space[2]]['std']]
                std_hyper = norm_jitter['std_hyper']
                distribution = norm_jitter['distribution'] #1.30添加，手工指定分布
                p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                
                transform_list += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space=color_space, distribution=distribution)]

            elif norm_jitter['color_space'] == 'Random': #1.10增加，混合多种方法，等概率随机进行选取
                distribution = norm_jitter['distribution'] #1.30添加，手工指定分布
                if 'L' in list(norm_jitter.keys()): #2.8修改，测试HED，lab，hsv三者的排列组合
                    mean_dataset = [norm_jitter['L']['avg'],norm_jitter['A']['avg'],norm_jitter['B']['avg']]
                    std_dataset = [norm_jitter['L']['std'],norm_jitter['A']['std'],norm_jitter['B']['std']]
                    std_hyper = norm_jitter['std_hyper']
                    p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                    transform_list += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space='LAB',distribution=distribution)]
                
                if 'E' in list(norm_jitter.keys()): #2.8修改，测试HED，lab，hsv三者的排列组合
                    mean_dataset = [norm_jitter['H']['avg'],norm_jitter['E']['avg'],norm_jitter['D']['avg']]
                    std_dataset = [norm_jitter['H']['std'],norm_jitter['E']['std'],norm_jitter['D']['std']]
                    std_hyper = norm_jitter['std_hyper']
                    p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                    # special_tfl += [hed_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=1)]
                    # 1.30修改，nj方法统一lab和hed，所以统一用一个即可
                    transform_list += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space='HED',distribution=distribution)]
                
                # 2.6修改，增加hsv来random
                if 'h' in list(norm_jitter.keys()): #2.8修改，测试HED，lab，hsv三者的排列组合
                    mean_dataset = [norm_jitter['h']['avg'],norm_jitter['S']['avg'],norm_jitter['V']['avg']]
                    std_dataset = [norm_jitter['h']['std'],norm_jitter['S']['std'],norm_jitter['V']['std']]
                    std_hyper = norm_jitter['std_hyper']
                    p = norm_jitter['p'] #2.9添加，采用增强的概率，默认是1
                    transform_list += [color_norm_jitter(mean=mean_dataset,std=std_dataset,std_hyper=std_hyper,probability=p,color_space='HSV',distribution=distribution)]
                    
    ###### baseline ###########
    if args.color_jitter is not None:
        brightness = args.color_jitter[0]
        contrast = args.color_jitter[1]
        saturation = args.color_jitter[2]
        hue = args.color_jitter[3]
        
        transform_list+=[transforms.RandomApply([transforms.ColorJitter(brightness, contrast, saturation, hue)],p=args.cj_p)]
                              
    if args.hed_jitter is not None:
        transform_list+=[transforms.RandomApply([HEDJitter(args.hed_jitter)],p=args.cj_p)]
    if args.lab_jitter is not None:
        if len(args.lab_jitter) == 1:
            transform_list+=[transforms.RandomApply([LABJitter(args.lab_jitter[0])],p=args.cj_p)]
        else:
            l_factor = args.lab_jitter[0]
            a_factor = args.lab_jitter[1]
            b_factor = args.lab_jitter[2]
            transform_list+=[transforms.RandomApply([LABJitter_hsv(l_factor,a_factor,b_factor)],p=args.cj_p)]
    
                    
    transform_list += [transforms.ToTensor()]
    mean_ = (0.485, 0.456, 0.406)
    std_ = (0.229, 0.224, 0.225)
    transform_list += [transforms.Normalize(
                        mean=torch.tensor(mean_),
                        std=torch.tensor(std_))
                      ]
    transform_train = transforms.Compose(transform_list)
    
    ###### test #######

    transform_test = transforms.Compose([
                    # transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean_),
                        std=torch.tensor(std_)
                    )
                ])
    
    print('train_transform:\n',transform_train)
    print('test_transform:\n',transform_test)
    
    train_dataset = ciaData(train_path,transform=transform_train)
    if args.nj_dynamic: #2.26加入dynamic
        _logger.info('nj_dynamic!!')
        train_dataset_copy = copy.deepcopy(train_dataset)
        train_dataset_copy.transform=transform_test
        train_dataset = ConcatDataset(train_dataset, train_dataset_copy)
    
    if not os.path.exists(val_path):
        val_dataset = ciaData(train_path,transform=transform_test)
    else:
        val_dataset = ciaData(val_path,transform=transform_test)
    
    test_dataset = ciaData(test_path,transform=transform_test)
    if args.nj_TTA > 0 : #2.26加入测试时增强
        _logger.info('nj_TTA {}!!'.format(args.nj_TTA))
        test_dataset_list = []
        for idx in range(args.nj_TTA+1): #多一个test_transform
            if idx == 0:
                test_dataset_copy = copy.deepcopy(test_dataset)
                test_dataset_list.append(test_dataset_copy)
            else:
                test_dataset_copy = copy.deepcopy(test_dataset)
                test_dataset_copy.transform=transform_train
                test_dataset_list.append(test_dataset_copy)
        # test_dataset = ConcatDataset(test_dataset_copy, test_dataset, test_dataset_copy) #只能用这种方法传入，不能一个完整list
        test_dataset_tuple = tuple(test_dataset_list)
        test_dataset = ConcatDataset(*test_dataset_tuple) #这样就可以取得类似的效果

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=15, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=15, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=15, pin_memory=True)

    model = CIAnet(growthRate=6, nDenseBlocks=[6,12,24,16], reduction=0.5, bottleneck=True).to(device)
    #checkpoint = torch.load('weights/CIA1.ptf')
    #model.load_state_dict(checkpoint['model_state_dict'])
    lr = args.lr #1e-5 #1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95, last_epoch=-1) #0.95
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20],gamma = 0.1)

    # Mask acc : 0.9201083183288574 --  Boundary acc : 0.5252535939216614
    # torch.save({'model_state_dict': model.state_dict()}, 'weights/CIA_initial.ptf')


    print('Begin training.')
    _logger.info('Begin training.')
    start = time.time()

    # loss_f, macc_f, bacc_f = [], [], []
    
    best_dict = {}
    best_dict['Dice'] = 0
    best_dict['Iou'] = 0
    
    Dynamic_P = Dynamic_P_class(epochs=args.epochs, batch_per_epoch=int(len(train_dataset)/args.batch_size)+1, dynamic_factor=args.dynamic_factor) #2.9新加，传递函数为引用类型即可
        
    for ep in range(epoch):
        ep += 1 
        save_idx = 0 # 每个epoch在test时保存一张图片
        # w = 0.75 - np.exp(2*ep/epoch)/(2*np.exp(2))
        # 2.22发现bug，ep较小的时候，w是负值
        # if ep < 35:
        #     w = 0.8
        # else:
        #     w = 0.2
        w = 0.8 #0.5
        train_bar = tqdm(train_loader)
        
        for batch in train_bar:
            model.train()
            if args.nj_dynamic == False: #2.26加入dynamic
                img, label, bound = batch
            else:
                batch_1, batch_2 = batch
                dynamic_p = Dynamic_P.step() #2.9新加全局函数，用step方法维护概率p，每个batch调整一次
                if np.random.rand(1) < dynamic_p: #选择有nj的
                    img, label, bound = batch_1
                else:   # 选择没有nj的
                    img, label, bound = batch_2
            img = img.to(device)
            label = label.to(device)
            bound = bound.to(device)
            mout, bout = model(img)
            # print(mout.shape, bout.shape)
            #loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, ep/(2*epoch)+1/4)
            loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound)
            # print(my_loss(mout, label), cia_loss(bout, bound, 0.5))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_description('Train Epoch: [{}/{}], lr: {:.8f}, Loss: {:.4f}'.format(ep, epoch, optimizer.param_groups[0]['lr'], loss))
            
        lr_scheduler.step() #每个epoch更新一次lr

        if ep % 1 == 0:
            # lr = lr * 0.99
            acc_all, iou_all, bacc_all, loss_all = [], [], [], []
            with torch.no_grad():
                for verify in tqdm(val_loader): #2.22修改，直接进行测试
                    img, label, bound = verify
                    if save_idx == 0 :
                        img_save = label.cpu().clone()
                        img_save = img_save.squeeze(0) # 压缩一维
                        img_save = transforms.ToPILImage()(img_save) # 自动转换为0-255
                        img_save.save(os.path.join(o_dir,'val/label_val.png'))
                    img = img.to(device)
                    label = label.to(device)
                    bound = bound.to(device)
                    model.eval()
                    mout, bout = model(img)
                    if save_idx == 0 :
                        img_save = mout[0][0].cpu().clone()
                        img_save = img_save.squeeze(0) # 压缩一维
                        img_save = transforms.ToPILImage()(img_save) # 自动转换为0-255
                        img_save.save(os.path.join(o_dir,'val/ep:{}-val.png'.format(ep)))
                        save_idx += 1
                    loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0.5)
                    loss_all.append(loss.cpu().numpy()) 
                    acc = dice_acc(mout[0][0], label) #没问题，这边acc计算默认b=1了
                    acc_all.append(acc)
                    acc = my_acc(mout[0][0], label[0][0])
                    if math.isnan(float(acc)):
                        pass
                    else:
                        iou_all.append(acc)
                    acc = dice_acc(bout[0][0], bound>0.1)
                    bacc_all.append(acc)

                acc_all = np.array(acc_all)
                iou_all = np.array(iou_all)
                loss_all = np.array(loss_all)
                bacc_all = np.array(bacc_all)
                
                _logger.info('epoch num val: {} -- Loss: {} -- Dice : {} -- Iou : {} --  Boundary acc : {}'.format(ep , round(float(loss_all.mean()), 4), round(float(acc_all.mean()),4), round(float(iou_all.mean()),4),round(bacc_all.mean(),4)))
                print('epoch num val: {} -- Loss: {} -- Dice : {} -- Iou : {} --  Boundary acc : {}'.format(ep , round(float(loss_all.mean()), 4), round(float(acc_all.mean()),4), round(float(iou_all.mean()),4),round(bacc_all.mean(),4)))

                # if ep > 49 and ep % 20 == 0:
                #     lr = lr * 0.95
                #     torch.save({'model_state_dict': model.state_dict()}, 'weights/ep{}_loss{}.ptf'.format(ep+1,bacc_all.mean()))

                # if ep % 2 == 0:
                #     torch.save({'model_state_dict': model.state_dict()}, '{}/ep{}_loss{}.ptf'.format(w_dir,ep,round(bacc_all.mean(), 3)))

            acc_all, iou_all, bacc_all, loss_all = [], [], [], []
            with torch.no_grad():
                for verify in tqdm(test_loader):
                    if args.nj_TTA > 0:
                        img, label, bound = verify[0]
                        img_list = [img.to(device)] #原图也搞上
                        for idx in range(args.nj_TTA):
                            img_list += [verify[idx+1][0].to(device)] #取第idx次增强的img
                    else:
                        img, label, bound = verify
                        
                    img = img.to(device)
                    label = label.to(device)
                    bound = bound.to(device)
                    
                    if save_idx == 1 :
                        img_save = label[0][0].cpu().clone()
                        img_save = img_save.squeeze(0) # 压缩一维
                        img_save = transforms.ToPILImage()(img_save) # 自动转换为0-255
                        img_save.save(os.path.join(o_dir,'test/label_test.png'))
                    
                    model.eval()
                    if args.nj_TTA > 0:
                        mout_mean = 0
                        bout_mean = 0
                        for idx in range(args.nj_TTA+1):
                            mout, bout = model(img_list[idx])
                            mout_mean = (mout + mout_mean * idx)/(idx+1)
                            bout_mean = (bout + bout_mean * idx)/(idx+1)
                        a = torch.ones_like(mout)
                        b = torch.zeros_like(mout)
                        thresh = 0.8
                        mout = torch.where(mout_mean>=thresh, a, b)
                        bout = torch.where(bout_mean>=thresh, a, b)
                    else:
                        mout, bout = model(img)
                    if save_idx == 1 :
                        img_save = mout[0][0].cpu().clone()
                        img_save = img_save.squeeze(0) # 压缩一维
                        img_save = transforms.ToPILImage()(img_save) # 自动转换为0-255
                        img_save.save(os.path.join(o_dir,'test/ep:{}-test.png'.format(ep)))
                        save_idx += 1
                    loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0.5)
                    loss_all.append(loss.cpu().numpy())
                    acc = dice_acc(mout[0][0], label)
                    acc_all.append(acc)
                    acc = my_acc(mout[0][0], label[0][0])
                    if math.isnan(float(acc)):
                        pass
                    else:
                        iou_all.append(acc)
                    acc = dice_acc(bout[0][0], bound>0.1)
                    bacc_all.append(acc)

                acc_all = np.array(acc_all)
                iou_all = np.array(iou_all)
                loss_all = np.array(loss_all)
                bacc_all = np.array(bacc_all)
                
                _logger.info('epoch num test: {} -- Loss: {} -- Dice : {} -- Iou : {} --  Boundary acc : {}\n'.format(ep , round(float(loss_all.mean()), 4), round(float(acc_all.mean()),4), round(float(iou_all.mean()),4),round(bacc_all.mean(),4)))
                print('epoch num test: {} -- Loss: {} -- Dice : {} -- Iou : {} --  Boundary acc : {}\n'.format(ep , round(float(loss_all.mean()), 4), round(float(acc_all.mean()),4), round(float(iou_all.mean()),4),round(bacc_all.mean(),4)))
                
                if best_dict['Dice'] < float(acc_all.mean()):
                    best_dict['Dice'] = round(float(acc_all.mean()),4)
                    best_dict['Iou'] = round(float(iou_all.mean()),4)
                    best_dict['epoch'] = ep
                    json_str = json.dumps(best_dict)
                    
                json_path = '{}/best.json'.format(o_dir)
                with open(json_path, 'w') as json_file:
                    json_file.write(json_str)
        
        #2.9每个epoch训练后，监视一下p的值
        if args.nj_dynamic != False:    
            print('dynamic_p',dynamic_p) 
    # macc_f = np.array(macc_f)
    # loss_f = np.array(loss_f)
    # bacc_f = np.array(bacc_f)
    # cacc_f = np.array(cacc_f)
    # mdic = {"macc":macc_f, "loss":loss_f,"bacc":bacc_f, "cacc":cacc_f}
    # scipy.io.savemat("results/cl_train.mat", mdic)
    
    torch.save({'model_state_dict': model.state_dict()}, 'weights/CIA.ptf')
    end = time.time()
    print('Total training time is {}h'.format((end-start)/3600))
    print('Finished Training')


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