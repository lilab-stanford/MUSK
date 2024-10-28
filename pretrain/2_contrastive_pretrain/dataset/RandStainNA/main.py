import os
from randstainna import RandStainNA
import cv2

if __name__ == "__main__":

    """
    Usage1: Demo(for visualization)
    """
    # # Setting: is_train = False
    # randstainna = RandStainNA(
    #     yaml_file = './randstainna/CRC_LAB_randomTrue_n0.yaml',
    #     std_hyper = 0.0,
    #     distribution = 'normal',
    #     probability = 1.0,
    #     is_train = False
    # )
    # print(randstainna)

    # img_path_list = [
    #     './visualization/origin/TUM-AEPINLNQ.png',
    #     './visualization/origin/TUM-DFGFFNEY.png',
    #     './visualization/origin/TUM-EWFNFSQL.png',
    #     './visualization/origin/TUM-TCGA-CVATFAAT.png'
    # ]
    # save_dir_path = './visualization/randstainna'
    # if not os.path.exists(save_dir_path):
    #     os.mkdir(save_dir_path)

    # for img_path in img_path_list:
    #     img = randstainna(cv2.imread(img_path))
    #     save_img_path = save_dir_path + '/{}'.format(img_path.split('/')[-1])
    #     cv2.imwrite(save_img_path,img)

    """
    Usage2：torchvision.transforms (for training)
    """
    # Setting: is_train = True
    from torchvision import transforms

    #### calling the randstainna
    transforms_list = [
        RandStainNA(
            yaml_file="./CRC_LAB_randomTrue_n0.yaml",
            std_hyper=-0.3,
            probability=1.0,
            distribution="normal",
            is_train=True,
        )
    ]

    transforms.Compose(transforms_list)
