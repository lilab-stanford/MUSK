import os
import time
import cv2
# from color_norm.Reinhard_quick import reinhard_cn, reinhard_cn_temp
from Reinhard_quick import reinhard_cn, reinhard_cn_temp

# template_path = '/root//autodl-tmp/pycharm_project_colorNorm/color_norm/demo/other/TUM-AIQIMVKD_template.png'
template_path = '/root/autodl-tmp/pycharm_project_CA2.5/TUM-EWFNFSQL.png'
# template_path = '/root/autodl-tmp/pycharm_project_CA2.5/TUM-AEPINLNQ.png'
# template_path = '/root/autodl-tmp/pycharm_project_CA2.5/TUM-TCGA-CVATFAAT.png'
# template_path = '/root//autodl-tmp/nine_class/train_use_2w/TUM/TUM-ACKCWNDR.png'
# template_path = '/root//autodl-tmp/nine_class/train_use_2w/TUM/TUM-ADEMNHMK.png'

# path_dataset = '/autodl-tmp/nine_class/standard/train'
# path_dataset = '/autodl-tmp/nine_class/standard/val'
# path_dataset = '/autodl-tmp/nine_class/standard/test'

path_dataset_list = [
    '/root/autodl-tmp/MoNuSeg2018/standard/train',
    '/root/autodl-tmp/MoNuSeg2018/standard/test'
]

# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/train'
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/val' #/val_colorNorm
# target_path_dataset = '/autodl-tmp/nine_class/colornorm_hsv/test' #/test_colorNorm

target_path_dataset_list = [
    '/root/autodl-tmp/MoNuSeg2018/colornorm_hed/train',
    '/root/autodl-tmp/MoNuSeg2018/colornorm_hed/test'

]

for target_path_dataset in target_path_dataset_list:
    if not os.path.isdir(target_path_dataset):
        os.makedirs(target_path_dataset)

i = 0

for idx in range(len(path_dataset_list)):
    path_dataset = path_dataset_list[idx]
    target_path_dataset = target_path_dataset_list[idx]
    print(path_dataset)
    print(target_path_dataset)
    for class_dir in os.listdir(path_dataset):
        if class_dir in ['label','bound']:
            continue
        path_class = os.path.join(path_dataset,class_dir)
        target_path_class = os.path.join(target_path_dataset,class_dir)
        # print(target_path_class)
        if not os.path.isdir(target_path_class):
            os.makedirs(target_path_class)
        # print(path_class)
        t1 = time.time()
        for image in os.listdir(path_class):
            i += 1
            path_img = os.path.join(path_class,image)
            # print(path_img)
            save_path = os.path.join(target_path_class,image)
            # print(save_path)
            # img = cv2.imread(path_img)
            # 以后一定要记得.ipynb_checkpoint的存在+try很好用
            # try:
            img_colorNorm = reinhard_cn(path_img,template_path,save_path,isDebug=False, color_space='HED') # 1.10增加，对归一化颜色空间的选择
    #             img_colorNorm = reinhard_cn_temp(path_img, template_path, save_path, isDebug=False)
            # except:
            #     print('path_img:',path_img)
            #     print('save_path:',save_path)
            #     continue

            if i % 200 == 0:
                t3 = time.time()
                print('i:',i, 'time:',t3-t1)
            # break
        t2 = time.time()
        print(t2-t1)
        # break