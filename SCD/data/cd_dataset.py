from .transform import Transforms
from util.palette import Color2Index
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


def make_dataset(dir):
    img_paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names


def mask_to_boundary(mask, dilation_ratio=0.005):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

    return mask - mask_erode


def multi_class_gt_to_boundary(gt, dilation_ratio=0.005):
    gt = gt.long()
    one_hot_gt = torch.eye(2)[gt]
    boundary_list = [np.expand_dims(mask_to_boundary(np.array(one_hot_gt[:,:,i], dtype=np.uint8), dilation_ratio=dilation_ratio),axis=0) for i in range(2)]
    boundary = np.concatenate(boundary_list, axis=0).sum(axis=0).astype(np.uint8)
    boundary = torch.from_numpy(boundary)

    return boundary


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt

        self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'im1')
        self.t1_paths, self.fnames = sorted(make_dataset(self.dir1))

        self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'im2')
        self.t2_paths, _ = sorted(make_dataset(self.dir2))

        self.dir_label1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label1')
        self.label1_paths, _ = sorted(make_dataset(self.dir_label1))

        self.dir_label2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label2')
        self.label2_paths, _ = sorted(make_dataset(self.dir_label2))

        self.dataset_size = len(self.t1_paths)

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose([Transforms()])
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        t1_path = self.t1_paths[index]
        fname = self.fnames[index]
        img1 = Image.open(t1_path)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path)

        label1_path = self.label1_paths[index]
        label1 = Image.open(label1_path)
        label1 = Image.fromarray(Color2Index(self.opt.dataset, np.array(label1)))

        label2_path = self.label2_paths[index]
        label2 = Image.open(label2_path)
        label2 = Image.fromarray(Color2Index(self.opt.dataset, np.array(label2)))

        mask = np.array(label1)
        cd_label = np.ones_like(mask)
        cd_label[mask == 0] = 0
        cd_label = Image.fromarray(cd_label)

        if self.opt.phase == 'train':
            _data = self.transform(
                {'img1': img1, 'img2': img2, 'label1': label1, 'label2': label2, 'cd_label': cd_label})
            img1, img2, label1, label2, cd_label = _data['img1'], _data['img2'], _data['label1'], _data['label2'], \
                _data['cd_label']

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        label1 = torch.from_numpy(np.array(label1)).long()
        label2 = torch.from_numpy(np.array(label2)).long()

        cd_label = torch.from_numpy(np.array(cd_label))
        boundary_mask = multi_class_gt_to_boundary(cd_label, dilation_ratio=self.opt.dilation_ratio)
        boundary_label = torch.ones(cd_label.size()) * 255
        boundary_label = (boundary_label * (1 - boundary_mask) + cd_label * boundary_mask).type_as(cd_label)
        input_dict = {'img1': img1, 'img2': img2, 'cd_label': cd_label, 'boundary_mask': boundary_mask, 'boundary_label': boundary_label, 'label1': label1, 'label2': label2, 'fname': fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers)
                                                      )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
