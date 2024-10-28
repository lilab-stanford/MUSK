import PIL.Image as Image
import os
from torchvision import transforms as transforms

img_path_list = [
    "./visualization/origin/TUM-AEPINLNQ.png",
    "./visualization/origin/TUM-DFGFFNEY.png",
    "./visualization/origin/TUM-EWFNFSQL.png",
    "./visualization/origin/TUM-TCGA-CVATFAAT.png",
]

save_dir_path = "./visualization/stain_augmentation"
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)

if __name__ == "__main__":
    for img_path in img_path_list:
        image = transforms.ColorJitter(
            brightness=0.35, contrast=0.5, saturation=0.5, hue=0.5
        )(Image.open(img_path))
        save_img_path = save_dir_path + "/{}".format(img_path.split("/")[-1])
        image.save(save_img_path)
