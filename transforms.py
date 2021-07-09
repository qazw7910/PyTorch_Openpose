import torch
import torchvision.transforms.functional as F
from PIL import Image
import random


class GroupRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            img_group = [F.hflip(img) for img in img_group]

        return img_group


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        return [F.resize(img, self.size, self.interpolation) for img in img_group]


class GroupToTensor(object):
    def __call__(self, img_group):
        return [F.to_tensor(img) for img in img_group]


class GetDiffTensor(object):
    def __init__(self, num_segments):
        self.num_segments = num_segments

    def __call__(self, tensor_group):
        diff_group = []
        length = len(tensor_group) // self.num_segments

        for seg_id in range(self.num_segments):
            for p in range(length - 1):
                offset = seg_id * length + p
                diff_group.append((tensor_group[offset + 1] - tensor_group[offset]))

        return diff_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_group):
        tensor_group = [tensor / 255 for tensor in tensor_group]
        return [F.normalize(tensor, self.mean, self.std) for tensor in tensor_group]


class GroupScaleJitterCrop(object):
    def __init__(self, h_ratio, w_ratio):
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def __call__(self, img_group):
        w, h = img_group[0].size
        new_w = int(w * self.w_ratio[random.randint(0, len(self.w_ratio) - 1)])
        new_h = int(h * self.h_ratio[random.randint(0, len(self.h_ratio) - 1)])
        w_coord = random.randint(0, w - new_w)
        h_coord = random.randint(0, h - new_h)

        return [F.crop(img, h_coord, w_coord, new_h, new_w) for img in img_group]


class Stack(object):
    def __call__(self, tensor_group):
        return torch.cat(tensor_group, dim=0)


if __name__ == '__main__':
    from dataset import TSNDataSet
    from preprocess.preprocess import read_info_file

    info_list = read_info_file('extract\\info.txt')
    dataSet = TSNDataSet(info_list=info_list, data_root='extract',
                         num_segments=10, new_length=1, modality='RGB')

    img_list, label = dataSet[11]
    img_list[1].save('ori.jpg')
    trans1 = GroupScaleJitterCrop([1., .9, .8], [1., .9, .8])

    out = trans1(img_list)
    out[1].save('rcrop.jpg')





