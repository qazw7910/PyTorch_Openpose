import torch
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

from keypoint_models.models import *
from keypoint_models.dataset import BodyKeypointsDataset
from keypoint_models.transform import *
from preprocess.BODY_25 import BODY_25


def get_point_xy_index(idx):
    return idx * 2, idx * 2+1


def draw_body_points(features, img_size=(640, 640, 3), zoom=None, shift=torch.Tensor([0, 0])):
    img = np.zeros(img_size, dtype=np.uint8)

    pairs = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6)]
    features = features.view(-1, 2)
    features_copy = features.clone().detach()

    if zoom:
        center = torch.FloatTensor([0.5, 0.5])
        features_copy = (features_copy - center) * zoom + center

    features_copy += shift

    for pair in pairs:
        pt1 = (int(features_copy[pair[0]][0] * img_size[0]), int(features_copy[pair[0]][1] * img_size[0]))
        pt2 = (int(features_copy[pair[1]][0] * img_size[1]), int(features_copy[pair[1]][1] * img_size[1]))
        cv2.line(img, pt1, pt2, (0, 255, 255), 5)

    for i in range(len(features_copy)):
        cv2.circle(
            img, (int(features_copy[i][0] * img_size[0]), int(features_copy[i][1] * img_size[1])), 5, (0, 0, 255), -1
        )

    return img


keypoints = [BODY_25.Nose, BODY_25.Neck, BODY_25.RShoulder, BODY_25.RElbow, BODY_25.RWrist,
             BODY_25.LShoulder, BODY_25.LElbow, BODY_25.LWrist,BODY_25.MidHip,
             BODY_25.RHip, BODY_25.RKnee, BODY_25.RAnkle, BODY_25.LHip, BODY_25.LKnee,
             BODY_25.LAnkle]

keypoints = [point.value for point in keypoints]
num_feature = 2 * len(keypoints)

torch.manual_seed(0)

dataset = BodyKeypointsDataset(keypoints, root_dir='data/', timesteps=450,
                               pad_by_last=True)
#改20
data = dataset[9]
X = data['feature']
pi = math.acos(-1)

zv = [1 + 0.5 * math.sin(i * pi / 90) for i in range(len(X))]
zv2 = [1 + 0.5 * math.sin(i * pi / 70) for i in range(len(X))]
sv = [0.3 * math.sin(i * pi / 90) for i in range(len(X))]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('combine.mp4', fourcc, 30, (640, 640))

for i in range(len(X)):
    # img = draw_body_points(X[i], zoom=zv[i])
    # img = draw_body_points(X[i], shift=torch.tensor([sv[i] * (i // 180 % 2), sv[i] * (1 - (i // 180 % 2))]))
    img = draw_body_points(X[i], zoom=zv2[i], shift=torch.tensor([sv[i] * (i // 180 % 2), sv[i] * (1 - (i // 180 % 2))]))
    writer.write(img)



