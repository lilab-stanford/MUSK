import os
import glob
import random

import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import pandas as pd
import h5py
import numpy
import json
import pandas as pd
from pathlib import Path
import io


class PCamDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        if train:
            self.data_x = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_train_x.h5"), "r")['x']
            self.data_y = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_train_y.h5"), "r")['y']
        else:
            self.data_x = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_test_x.h5"), "r")['x']
            self.data_y = h5py.File(os.path.join(root, "camelyonpatch_level_2_split_test_y.h5"), "r")['y']

        self.trans = transform

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]
        self.classes = ['Lymph node', 'Lymph node containing metastatic tumor tissue']

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, item):
        image = Image.fromarray(self.data_x[item])
        label = self.data_y[item]

        if self.trans is not None:
            if self.trans.__class__.__name__ == "CLIPProcessor":
                image = self.trans(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.trans(image)

        label = int(numpy.squeeze(label))
        return image, label




class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transform=None, train=True, val=False,
                 tumor=False):
        csv_file = os.path.join(root, csv_file)
        self.data = pd.read_csv(csv_file)
        self.data_root = root

        if train:
            self.data = self.data[self.data['set'] == 'Train']
        else:
            if val:
                self.data = self.data[self.data['set'] == "Validation"]
            else:
                self.data = self.data[self.data['set'] == 'Test']

        if tumor:
            self.data = self.data[self.data['malignicy'] == 'tumor']
        self.tumor = tumor

        self.image_paths = self.data['file'].values
        self.labels = self.data['class'].values

        self.transform = transform
        self.train = train

        self.cat_to_num_map = {'nontumor_skin_necrosis_necrosis': 0,
                               'nontumor_skin_muscle_skeletal': 1,
                               'nontumor_skin_sweatglands_sweatglands': 2,
                               'nontumor_skin_vessel_vessel': 3,
                               'nontumor_skin_elastosis_elastosis': 4,
                               'nontumor_skin_chondraltissue_chondraltissue': 5,
                               'nontumor_skin_hairfollicle_hairfollicle': 6,
                               'nontumor_skin_epidermis_epidermis': 7,
                               'nontumor_skin_nerves_nerves': 8,
                               'nontumor_skin_subcutis_subcutis': 9,
                               'nontumor_skin_dermis_dermis': 10,
                               'nontumor_skin_sebaceousglands_sebaceousglands': 11,
                               'tumor_skin_epithelial_sqcc': 12,
                               'tumor_skin_melanoma_melanoma': 13,
                               'tumor_skin_epithelial_bcc': 14,
                               'tumor_skin_naevus_naevus': 15
                               }

        self.tumor_map = {'tumor_skin_epithelial_sqcc': 0,
                          'tumor_skin_melanoma_melanoma': 1,
                          'tumor_skin_epithelial_bcc': 2,
                          'tumor_skin_naevus_naevus': 3
                          }

        self.classes = list(self.cat_to_num_map) if not self.tumor else list(self.tumor_map)

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(os.path.join(self.data_root, image_path)).convert('RGB')

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        if not self.tumor:
            label = self.cat_to_num_map[self.labels[index]]
        else:
            label = self.tumor_map[self.labels[index]]

        return image, label


class PannukeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.root = root

        df = pd.read_csv(os.path.join(root, "PanNuke_all_binary.csv"))
        self.df = df[df['split'] == 'train'] if train else df[df['split'] == 'test']

        self.transform = transform

        self.classes = ["benign",
                        "malignant"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.df.iloc[index]['image'])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = 1 if 'malignant' in self.df.iloc[index]['caption'] else 0
        return image, label



class UnitopathoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        if train:
            self.data = json.load(open(os.path.join(root, "images_train.json")))
        else:
            self.data = json.load(open(os.path.join(root, "images_test.json")))
        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}
        # NORM - Normal
        # tissue;
        # HP - Hyperplastic
        # Polyp;
        # TA.HG - Tubular
        # Adenoma, High - Grade
        # dysplasia;
        # TA.LG - Tubular
        # Adenoma, Low - Grade
        # dysplasia;
        # TVA.HG - Tubulo - Villous
        # Adenoma, High - Grade
        # dysplasia;
        # TVA.LG - Tubulo - Villous
        # Adenoma, Low - Grade
        # dysplasia.

        self.classes = ["Hyperplastic Polyp",
                        "Normal tissue",
                        "Tubular Adenoma, High-Grade dysplasia",
                        "Tubular Adenoma, Low-Grade dysplasia",
                        "Tubulo-Villous Adenoma, High-Grade dysplasia",
                        "Tubulo-Villous Adenoma, Low-Grade dysplasia"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label


class UnitopathoRetrievalDataset(torch.utils.data.Dataset):
    """
    Dataset for unitopatho image retrieval, using all samples.
    """

    def __init__(self, root, transform=None, train=True):
        fp1 = json.load(open(os.path.join(root, "images_train.json")))
        fp2 = json.load(open(os.path.join(root, "images_test.json")))
        self.data = fp1 + fp2

        self.root = root
        self.transform = transform

        self.labels_dict = {"HP": 0,
                            "NORM": 1,
                            "TA.HG": 2,
                            "TA.LG": 3,
                            "TVA.HG": 4,
                            "TVA.LG": 5}

        # these prompts work better!
        self.classes = ["HP",
                        "NORM",
                        "TA.HG",
                        "TA.LG",
                        "TVA.HG",
                        "TVA.LG"]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = os.path.join(self.root, self.data[index])
        image = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[fpath.split("/")[-2]]

        return image, label



class BRACS6ClsDataset(torch.utils.data.Dataset):
    """
    BRACS dataset ROIs are scanned in 40x; and the image size is around 1800*1800 - 2400*2400;
    we get 224*224 at 5x; thus, the crop_size = 224*8.
    """
    def __init__(self, root, transform=None, train=True, is_retrieval=False):

        self.transform = transform
        self.root = root

        self.crop_size = 224 * 8
        df = pd.read_csv(os.path.join(root, "bright_roi.csv"))

        if not is_retrieval:
            # take data split for classification task
            if train:
                self.df = df[df['data_split'] == 'train']
            else:
                self.df = df[df['data_split'] == 'test']
        else:
            # use all images for image retrieval task
            self.df = df

        self.labels_dict = {"0_N": 0,
                            "1_PB": 0,
                            "2_UDH": 1,
                            "3_FEA": 2,
                            "4_ADH": 3,
                            "5_DCIS": 4,
                            "6_IC": 5
                            }

        self.classes = ["normal tissue or pathological benign",
                        "Usual Ductal Hyperplasia ",
                        "Flat Epithelia Atypia ",
                        "Atypical Ductal Hyperplasia",
                        "Ductal Carcinoma in Situ",
                        "Invasive Carcinoma"
                        ]

        self.templates = ["a histopathology slide showing {c}",
                          "histopathology image of {c}",
                          "pathology tissue showing {c}",
                          "presence of {c} tissue on image"]

    def __len__(self):
        return len(self.df)

    def crop_or_pad_image(self, im):
        desired_size = self.crop_size
        old_size = im.size

        if old_size[0] > desired_size or old_size[1] > desired_size:
            # Crop the image
            left = (old_size[0] - desired_size) / 2
            top = (old_size[1] - desired_size) / 2
            right = (old_size[0] + desired_size) / 2
            bottom = (old_size[1] + desired_size) / 2

            im = im.crop((left, top, right, bottom))
        else:
            # Pad the image
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size - old_size[0]) // 2, (desired_size - old_size[1]) // 2))
            im = new_im

        return im

    def __getitem__(self, index):
        while True:
            try:
                item = self.df.iloc[index]
                image_path = os.path.join(self.root, item.image_path)
                image = Image.open(image_path).convert("RGB")
                image = self.crop_or_pad_image(image)
                break
            except:
                print(f"{item.image_path} failed.")
                os.system(f"rm -rf {item.image_path}")
                index = random.randint(0, self.__len__() - 1)

        if self.transform is not None:
            if self.transform.__class__.__name__ == "CLIPProcessor":
                image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                image = self.transform(image)

        label = self.labels_dict[item.class_name]

        return image, label


class BRACS3ClsDataset(BRACS6ClsDataset):
    """
    BRACS dataset ROIs are scanned in 40x; and the image size is around 1800*1800 - 2400*2400;
    we get 224*224 at 5x; thus, the crop_size = 224*8.
    """
    def __init__(self, root, transform=None, train=True, is_retrieval=False):
        super(BRACS3ClsDataset, self).__init__(root, transform, train, is_retrieval)

        self.labels_dict = {"0_N": 0,
                            "1_PB": 0,
                            "2_UDH": 0,
                            "3_FEA": 1,
                            "4_ADH": 1,
                            "5_DCIS": 2,
                            "6_IC": 2
                            }

        self.classes = ["Non-cancerous	",
                        "Pre-cancerous	 ",
                        "Cancerous "
                        ]



class PathMMUDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """
        Initialize the dataset by processing the JSON files and setting up the image roots.
        """
        self.transform = transform
        self.root = Path(root)
        self.items = []

        # Process subsets
        self.img_root = {
            "Book": self.root / "Book/images",
            "WebPathology": self.root / "WebPathology/images",
            "Twitter": self.root / "Twitter/images",
        }

        self._process_json_files(self.root / "Book")
        self._process_json_files(self.root / "WebPathology")
        self._process_json_files(self.root / "Twitter")

    def _process_json_files(self, dir_path):
        """
        Process all JSON files in the given directory and extend the items list.
        """
        dir_path = dir_path / "GPTCaption"
        json_files = dir_path.glob("*.json")
        for json_file in json_files:
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    
                    data_list = list(data.values())
                    dataset_name = json_file.parts[-3]
                    for entry in data_list:
                        entry['dataset'] = dataset_name

                    self.items.extend(data_list)

                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {json_file}")

    def __len__(self):
        """
        Return the total number of items in the dataset.
        """
        return len(self.items)

    def __getitem__(self, idx):
        """
        Retrieve an item by index, including the image and its corresponding caption.
        """
        
        item = self.items[idx]
        img_root = self.img_root[item['dataset']]
        img_path = img_root / item['img_path']

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image file not found: {img_path}")
            return None
        
        # for clip model
        if self.transform.__class__.__name__ == "CLIPProcessor":
            image = self.transform(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        # for other models
        else:
            image = self.transform(image)
        
        return image, item['caption']
