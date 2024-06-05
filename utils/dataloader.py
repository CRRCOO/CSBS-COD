import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.data_augmentation import cv_random_flip, randomCrop, randomRotation, randomPeper, colorEnhance
from config import DataPath


class TrainDataset(Dataset):
    """
    dataloader for COD tasks
    Implemented according to DGNet
    """
    def __init__(self, image_root, gt_root, trainsize, edge_root=None):
        self.edge_root = edge_root
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        if edge_root is not None:
            self.edges = [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith('.jpg')
                          or f.endswith('.png')]

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        # filter mathcing degrees of files
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        # Data Augmentation
        # random horizental flipping
        if self.edge_root is not None:
            image, gt, edge = cv_random_flip([image, gt, edge])
            image, gt, edge = randomCrop([image, gt, edge])
            image, gt, edge = randomRotation([image, gt, edge])
        else:
            image, gt = cv_random_flip([image, gt])
            image, gt = randomCrop([image, gt])
            image, gt = randomRotation([image, gt])
        # bright, contrast, color, sharp jitters
        image = colorEnhance(image)
        # random peper noise
        gt = randomPeper(gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        if self.edge_root is not None:
            edge = self.gt_transform(edge)
        if self.edge_root is not None:
            return image, gt, edge
        else:
            return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        if self.edge_root is not None:
            assert len(self.edges) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, testsize, edge_root=None):
        self.testsize = testsize
        self.edge_root = edge_root
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        if edge_root is not None:
            self.edges = [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.endswith('.jpg')
                          or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        image = self.transform(image)
        gt_origin = transforms.PILToTensor()(gt)
        gt = self.gt_transform(gt)
        if self.edge_root is not None:
            edge_origin = transforms.PILToTensor()(edge)
            edge = self.gt_transform(edge)
        name = self.images[index].split('/')[-1]
        if '\\' in name:
            name = name.split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        if self.edge_root is not None:
            return image, gt, gt_origin, edge, edge_origin, name
        else:
            return image, gt, gt_origin, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
