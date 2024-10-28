from genericpath import exists
import os 
import cv2 
import numpy as np

def create_bound(path,out_path,name):
    size = 1000
    masks = os.listdir(os.path.join(path,name))
    bou = np.zeros([size, size,1], np.uint8)
    re = np.zeros([size, size,1], np.uint8)
    
    for i, mask in enumerate(masks):
        # if n in ['.DS_Store' , 'instance_mask']:
        #     continue
        img=cv2.imread(os.path.join(path, name, mask),cv2.IMREAD_GRAYSCALE)
        re = cv2.bitwise_or(re, img)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        img1=cv2.erode(img, kernel1, iterations=1)
        img2=cv2.dilate(img, kernel2, iterations=1)
        result=img2-img1
        bou=cv2.bitwise_or(bou,result)
    '''
    both = np.hstack((re, bou))
    cv2.imshow('imgc', both)
    cv2.waitKey(0)
    '''
    # _,name=os.path.split(path)
    print ('done',name)
    
    cv2.imwrite(os.path.join(out_path,'mask/{}.png'.format(name)), re)
    cv2.imwrite(os.path.join(out_path,'bound/{}.png'.format(name)), bou)

if __name__ == '__main__': 
    from tqdm import tqdm
    path = '../../images/MoNuSeg/MoNuSegTestData/instance_masks'
    out_path = '../../images/MoNuSeg/MoNuSegTestData/CA2_5'
    if not os.path.exists(os.path.join(out_path,'mask')):
        os.makedirs(os.path.join(out_path,'mask'))
    if not os.path.exists(os.path.join(out_path,'bound')):
        os.makedirs(os.path.join(out_path,'bound'))

    for name in tqdm(os.listdir(path)):
        if name in ['.DS_Store' , 'instance_mask']:
            continue

        create_bound(path,out_path,name)