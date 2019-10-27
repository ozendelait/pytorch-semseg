import os
import json
import torch
import numpy as np

from torch.utils import data
from PIL import Image

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate


class mapillaryVistasLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        img_size=(640, 1280),
        is_transform=True,
        augmentations=None,
        test_mode=False,
        version='cityscapes',
        asp_ratio_delta_min = -1.0,
        asp_ratio_delta_max = -1.0,  
        img_norm=True
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 65      
        self.img_norm = img_norm
        self.asp_ratio_delta_min = asp_ratio_delta_min
        self.asp_ratio_delta_max = asp_ratio_delta_max 

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([80.5423, 91.3162, 81.4312])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, "images")
        self.annotations_base = os.path.join(self.root, self.split, "labels")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        self.class_ids, self.class_names, self.class_colors = self.parse_config()

        self.ignore_id = 250
        self.first_run = True
        self.lut = None
        if version == 'cityscapes':
            map_to_cs = [250, 250,   1,   4,   4,   2,   3, 0,   1,   1,   0,   1, 250,
         0,   0,   1,   2,   2,   2,  11,  12,  12,  12,   0, 0,   8,
         9,  10,   9,   9,   8,   9,   2,   2,   2,   2, 250,   2,   5,
         2, 250, 250,   2, 250,   5,   5,   5,   5,   6,   5,   7,   2,
        18, 250,  15,  13,  14,  17,  16,  14,  14,  14,  11, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250,
        250, 250, 250, 250, 250, 250, 250, 250, 250]
            self.lut = np.array(map_to_cs).astype(np.uint8)
            self.n_classes = 19
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def parse_config(self):
        with open(os.path.join(self.root, "config.json")) as config_file:
            config = json.load(config_file)

        labels = config["labels"]

        class_names = []
        class_ids = []
        class_colors = []
        print("There are {} labels in the config file".format(len(labels)))
        for label_id, label in enumerate(labels):
            class_names.append(label["readable"])
            class_ids.append(label_id)
            class_colors.append(label["color"])

        return class_names, class_ids, class_colors

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")
        )
        
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        #print("INFO sz0", img_path, img.size, lbl.size, self.img_size)
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        if not self.lut is None:
            lbl = torch.from_numpy(np.take(self.lut, lbl)).long()
        #print("INFO sz1", self.is_transform, img_path, img.size, lbl.size, self.img_size)
        return img, lbl

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            asp_ratio_src = img.size[0]/img.size[1]
            asp_ratio_trg = self.img_size[1]/self.img_size[0]
            if self.asp_ratio_delta_min > 0.0 and asp_ratio_src < asp_ratio_trg*self.asp_ratio_delta_min:
                if img.size[0] >= img.size[1]:
                    w_trg = img.size[1]*asp_ratio_trg*self.asp_ratio_delta_min
                    img = img.crop(((img.size[0]-w_trg)/2, 0, w_trg, img.size[1]))
                else:
                    h_trg = img.size[0]/(asp_ratio_trg*self.asp_ratio_delta_min)
                    img = img.crop((0,(img.size[1]-h_trg)/2, img.size[0], h_trg))
            elif self.asp_ratio_delta_max > 0.0 and asp_ratio_src > asp_ratio_trg*self.asp_ratio_delta_max:
                if img.size[0] >= img.size[1]:
                    w_trg = img.size[1]*asp_ratio_trg*self.asp_ratio_delta_max
                    img = img.crop(((img.size[0]-w_trg)/2, 0, w_trg, img.size[1]))
                else:
                    h_trg = img.size[0]/(asp_ratio_trg*self.asp_ratio_delta_max)
                    img = img.crop((0,(img.size[1]-h_trg)/2, img.size[0], h_trg)) 
            #if self.first_run:
            #    print("Input0: self.img_size (h,w) "+str(self.img_size) + " img_sz (w,h): "+ str(img.size))
            self.first_run = False
            #trg_h_sc = self.img_size[1]
            img = img.resize(
                (self.img_size[1], self.img_size[0]), resample=Image.LANCZOS
            )  # uint8 with RGB mode
            lbl = lbl.resize((img.size[0], img.size[1]))
        img = np.array(img).astype(np.float64)
        if self.img_norm:
            img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # From HWC to CHW
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 65] = self.ignore_id
        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.class_colors[l][0]
            g[temp == l] = self.class_colors[l][1]
            b[temp == l] = self.class_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    augment = Compose([RandomHorizontallyFlip(), RandomRotate(6)])

    local_path = "/private/home/meetshah/datasets/seg/vistas/"
    dst = mapillaryVistasLoader(
        local_path, img_size=(512, 1024), is_transform=True, augmentations=augment
    )
    bs = 8
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=4, shuffle=True)
    for i, data_samples in enumerate(trainloader):
        x = dst.decode_segmap(data_samples[1][0].numpy())
        print("batch :", i)
