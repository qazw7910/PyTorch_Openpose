import torch
import numpy as np


class SkipByTimeStep(object):
    def __init__(self, timestep):
        self.timestep = timestep

    def __call__(self, data):
        feature = data['feature']
        seq_len = data['seq_len']

        num_blocks = len(feature) // self.timestep
        first_idx = np.random.randint(low=0, high=self.timestep)
        offsets = [self.timestep * i + first_idx for i in range(num_blocks)]

        new_feature = feature[offsets, :]

        new_seq_len = seq_len // self.timestep
        if (seq_len - 1) % self.timestep >= first_idx:
            new_seq_len += 1

        data['seq_len'] = new_seq_len
        data['feature'] = new_feature
        return data


class Centralize(object):
    def __call__(self, data):
        feature = data['feature']
        v_shift = 0.5 - feature[0, 0]
        h_shift = 0.5 - feature[0, 1]

        feature[:, 0::2] = feature[:, ::2] + v_shift
        feature[:, 1::2] = feature[:, 1::2] + h_shift

        data['feature'] = feature

        return data


class Scale(object):
    def __init__(self, down_ratio=1, up_ratio=1):
        self.down_ratio = down_ratio
        self.up_ratio = up_ratio

    def __call__(self, data):
        feature = data['feature']

        ratio = np.random.uniform(self.down_ratio, self.up_ratio)
        feature = 0.5 + (feature[:, :] - 0.5) * ratio
        data['feature'] = feature

        return data


class RandomShift(object):
    def __init__(self, LShift, RShift, DShift, UShift):
        self.LShift = LShift
        self.RShift = RShift
        self.UShift = UShift
        self.DShift = DShift

    def __call__(self, data):
        feature = data['feature']
        HShift = np.random.uniform(self.LShift, self.RShift)
        VShift = np.random.uniform(self.DShift, self.UShift)

        new_feature = torch.FloatTensor(feature)
        new_feature[:, 0::2] = feature[:, 0::2] + VShift
        new_feature[:, 1::2] = feature[:, 1::2] + HShift

        data['feature'] = new_feature
        return data

class NLCtoNCL(object):
    def __call__(self, data):
        new_feature = data['feature'].permute(1, 0).contiguous()
        data['feature'] = new_feature

        return data
