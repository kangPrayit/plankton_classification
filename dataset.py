import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms


class PlanktonDataset(Dataset):
    def __init__(self, data_dir, annotations_file='_annotations.coco.json', transform=None):
        super(PlanktonDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        with open(os.path.join(data_dir, annotations_file), 'r') as f:
            coco_annotations = json.load(f)

        self.plankton_images = coco_annotations['images']
        self.plankton_annotations = coco_annotations['annotations']
        self.plankton_categories = coco_annotations['categories']

    def __len__(self):
        return len(self.plankton_images)

    def __getitem__(self, item):
        plankton_img = self.plankton_images[item]['file_name']
        plankton_img = cv2.imread(os.path.join(self.data_dir, plankton_img))
        plankton_img = cv2.cvtColor(plankton_img, cv2.COLOR_BGR2RGB)
        mask = np.zeros(plankton_img.shape[:2], dtype=np.uint8)
        image_id = self.plankton_images[item]['id']
        image_annotations = [ann for ann in self.plankton_annotations if ann['image_id'] == image_id]
        for annotation in image_annotations:
            mask_annotations = annotation['segmentation']
            mask_annotations = np.array(mask_annotations, dtype=np.int32)
            mask_annotations = np.reshape(mask_annotations, (-1, 2))
            cv2.fillPoly(mask, [mask_annotations], 255)
        plankton_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        assert plankton_img.size == plankton_mask.size

        if self.transform:
            plankton_img = self.transform(plankton_img)
            plankton_mask = self.transform(plankton_mask)

        return plankton_img, plankton_mask


class PlanktonSegmentationDataset(Dataset):
    def __init__(self, data_dir, annotation_file='_annotations.coco.json', transform=None):
        super(PlanktonSegmentationDataset, self).__init__()
        self.data_dir = data_dir
        self.annotations = COCO(os.path.join(data_dir, annotation_file))
        self.image_ids = self.annotations.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_info = self.annotations.loadImgs(self.image_ids[item])[0]
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        annotation_ids = self.annotations.getAnnIds(imgIds=self.image_ids[item])
        annotations = self.annotations.loadAnns(annotation_ids)
        masks = [coco_mask.decode(self.annotations.annToRLE(ann)) for ann in annotations]
        target = torch.zeros((image.size[1], image.size[0]), dtype=torch.float32)
        for i, mask in enumerate(masks):
            target[mask == 1] = i + 1
        target = target.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, target


if __name__ == '__main__':
    # train_ds = PlanktonDataset(data_dir='./datasets/plankton_cocov2/train', transform=transforms.ToTensor())
    # img, mask = train_ds[30]
    # print(img.shape, mask.shape)
    # plt.imshow(img)
    # plt.show()
    #
    # plt.imshow(mask)
    # plt.show()

    train_ds = PlanktonSegmentationDataset(data_dir='./datasets/plankton_cocov2/train', transform=transforms.ToTensor())
    image, target = train_ds[11]
    print(image.shape, target.shape)
    # image.show()
    # target_img = transforms.ToPILImage()(target)
    # target_img.show()

