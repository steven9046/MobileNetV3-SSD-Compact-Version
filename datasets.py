import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        # print("getting image: %s"%self.images[i])
        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]
        # print("image size: ", image.size[0], image.size[1])
        # print("boxes before transform :", boxes.shape)#[3,4]
        # print("labels before transform :", labels.shape)#[3]
        # for i in range(boxes.size(0)):
        #     print("box: ")
        #     print(boxes[i, 0].item())
        #     print(boxes[i, 1].item())
        #     print(boxes[i, 2].item())
        #     print(boxes[i, 3].item())
        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        # print("image tensor shape:")
        # print(image.shape)
        # print("boxes after transform :", boxes.shape)#[3,4]
        # print("labels after transform :", labels.shape)#[3]
        # for i in range(boxes.shape[0]):
        #     print("box: ")
        #     print(boxes[i, 0].item())
        #     print(boxes[i, 1].item())
        #     print(boxes[i, 2].item())
        #     print(boxes[i, 3].item())

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        # print("arranging tensor shape....")
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()
        # 从dataset里得到一个batch的数据，每个batch里是dataset的get item返回的东西
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        # print("Done")
        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
