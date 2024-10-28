import PIL.Image as Image
import os
from torchvision import transforms as transforms
import cv2
import numpy as np
from skimage import color


def quick_loop(image, image_avg, image_std, temp_avg, temp_std, isHed=False):

    image = (image - np.array(image_avg)) * (
        np.array(temp_std) / np.array(image_std)
    ) + np.array(temp_avg)
    if isHed:  # HED in range[0,1]
        pass
    else:  # LAB/HSV in range[0,255]
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def getavgstd(image):
    avg = []
    std = []
    image_avg_l = np.mean(image[:, :, 0])
    image_std_l = np.std(image[:, :, 0])
    image_avg_a = np.mean(image[:, :, 1])
    image_std_a = np.std(image[:, :, 1])
    image_avg_b = np.mean(image[:, :, 2])
    image_std_b = np.std(image[:, :, 2])
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(image_std_l)
    std.append(image_std_a)
    std.append(image_std_b)
    return (avg, std)


def reinhard_cn(image_path, temp_path, save_path, isDebug=False, color_space=None):
    isHed = False
    image = cv2.imread(image_path)
    if isDebug:
        cv2.imwrite("source.png", image)
    template = cv2.imread(temp_path)  ### template images
    if isDebug:
        cv2.imwrite("template.png", template)

    if color_space == "LAB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # LAB range[0,255]
        template = cv2.cvtColor(template, cv2.COLOR_BGR2LAB)
    elif color_space == "HED":
        isHed = True
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # color.rgb2hed needs RGB as input
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

        image = color.rgb2hed(image)  # HED range[0,1]
        template = color.rgb2hed(template)
    elif color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    elif color_space == "GRAY":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path, image)
        return

    image_avg, image_std = getavgstd(image)
    template_avg, template_std = getavgstd(template)
    if isDebug:
        print("isDebug!!!")
        print("source_avg: ", image_avg)
        print("source_std: ", image_std)
        print("target_avg: ", template_avg)
        print("target_std: ", template_std)

    # Reinhard's Method to Stain Normalization
    image = quick_loop(
        image, image_avg, image_std, template_avg, template_std, isHed=isHed
    )

    if color_space == "LAB":
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(save_path, image)
    elif color_space == "HED":  # HED[0,1]->RGB[0,255]
        image = color.hed2rgb(image)
        imin = image.min()
        imax = image.max()
        image = (255 * (image - imin) / (imax - imin)).astype("uint8")
        image = Image.fromarray(image)
        image.save(save_path)
    elif color_space == "HSV":
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(save_path, image)

    if isDebug:
        cv2.imwrite("results.png", image)


if __name__ == "__main__":
    img_path_list = [
        "./visualization/origin/TUM-AEPINLNQ.png",
        "./visualization/origin/TUM-DFGFFNEY.png",
        "./visualization/origin/TUM-EWFNFSQL.png",
        "./visualization/origin/TUM-TCGA-CVATFAAT.png",
    ]
    template_path = "./visualization/origin/TUM-EWFNFSQL.png"
    save_dir_path = "./visualization/stain_normalization"
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    for img_path in img_path_list:
        save_path = save_dir_path + "/{}".format(img_path.split("/")[-1])
        img_colorNorm = reinhard_cn(
            img_path, template_path, save_path, isDebug=False, color_space="LAB"
        )
