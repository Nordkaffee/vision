""" Coffea Arabica Dataset

This module contains a dataset class which loads the images of Coffea Arabica
cherries together with the matching bounding boxes.

"""

from typing import List
from PIL import Image

import os
import json
import random

import torch
import torch.nn as nn

import torchvision


def _get_label_by_filename(fname: str, labels: List[dict]):
    """

    """
    return next(l for l in labels if fname in l['External ID'])


def _parse_json_label(label: dict, index: int):
    """

    """

    #
    boxes = []
    labels = []
    for i, box in enumerate(label['Label']['objects']):
        xmin = box['bbox']['left']
        xmax = xmin + box['bbox']['width']
        ymin = box['bbox']['top']
        ymax = ymin + box['bbox']['height']
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1 if box['title'] == 'Cherry (ripe)' else 2)

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    #
    image_id = torch.tensor([index])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    #
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    return target


def _horizontal_flip(img: Image, target: dict):
    """

    """
    img = torchvision.transforms.functional.hflip(img)

    w, h = img.size
    for i, box in enumerate(target['boxes']):
        xmin = w - box[2] # w - xmax
        xmax = w - box[0] # w - xmin
        ymin = box[1]
        ymax = box[3]
        assert xmax > xmin
        assert ymax > ymin
        target['boxes'][i] = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

    return img, target


def _vertical_flip(img: Image, target: dict):
    """

    """
    img = torchvision.transforms.functional.vflip(img)

    w, h = img.size
    for i, box in enumerate(target['boxes']):
        xmin = box[0]
        xmax = box[2]
        ymin = h - box[3] # h - ymax
        ymax = h - box[1] # h - ymin
        assert xmax > xmin
        assert ymax > ymin
        target['boxes'][i] = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

    return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


class Dataset(torchvision.datasets.ImageFolder):
    """

    """

    def __init__(self, input_dir: str, train: bool = False):

        # load images
        self.input_dir = input_dir
        super(Dataset, self).__init__(input_dir)

        # load labels
        with open(os.path.join(input_dir, 'labels.json'), 'r') as json_file:
            json_labels = json.load(json_file)
 
        self.index_to_labels = {}
        for index, (fpath, _) in enumerate(self.samples):
            fname = os.path.basename(fpath)
            label = _get_label_by_filename(fname, json_labels)
            label = _parse_json_label(label, index)
            self.index_to_labels[index] = label

        self.train = train
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.1,
        )
        self.gaussian_blur = torchvision.transforms.GaussianBlur(21)
        self.to_tensor = torchvision.transforms.ToTensor()


    def __getitem__(self, idx):
        """

        """
        img, _ = super(Dataset, self).__getitem__(idx)
        target = self.index_to_labels[idx]

        if self.train:
            # horizontal flip
            if random.random() < 0.5:
                img, target = _horizontal_flip(img, target)
            if random.random() < 0.5:
                img, target = _vertical_flip(img, target)
            if random.random() < 0.0:
                img = self.gaussian_blur(img)            

            img = self.color_jitter(img)
        
        img = self.to_tensor(img)

        return img, target


if __name__ == '__main__':

    dataset = Dataset('datasets/initial-dataset-labeled', True)
    print(f'Found {len(dataset)} images.')
    img, target = dataset[11]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=collate_fn,
    )

    images, targets = next(iter(dataloader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]

    from model import CoffeAI
    coffeai = CoffeAI.from_checkpoint('checkpoints/pretrained.pt')
    print(coffeai(images, targets))
