from torch.utils.data import Dataset
import torch
import os
import sys
import numpy as np


class BodyKeypointsDataset(Dataset):

    def __init__(self, keypoints, root_dir, timesteps, pad_by_last=False, transforms=None):
        self.num_feature = len(keypoints) * 2
        self.phase = None
        self.timesteps = timesteps
        self.transforms = transforms
        self.data, self.labels = self._read_data(keypoints, root_dir)   #_get_label, _read_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret_data = torch.tensor(self.data[idx]).float()
        ret_label = torch.tensor(self.labels[idx]).long()

        if self.phase == 'train':
            indices = self._random_sample_indices(len(ret_data), self.timesteps)
        elif self.phase == 'val' or 'test':  #new test_dataset
            indices = self._sample_middle_indices(len(ret_data), self.timesteps)
        else:
            indices = list(range(len(ret_data)))

        ret_data = ret_data[indices]

        item = {'feature': ret_data, 'label': ret_label}

        if self.transforms is not None and self.phase is not None:
            item = self.transforms[self.phase](item)

        return item

    def set_phase(self, phase):
        self.phase = phase

    def _random_sample_indices(self, total_length, timesteps):
        block_size = total_length // timesteps
        last_block_size = total_length - (timesteps - 1) * block_size
        offsets = np.random.randint(0, block_size, size=timesteps-1)
        offsets = np.append(offsets, np.random.randint(0, last_block_size, size=1))
        indices = [block_size * i + offset for i, offset in enumerate(offsets)]

        return indices

    def _sample_middle_indices(self, total_length, timesteps):
        block_size = total_length // timesteps
        last_block_size = total_length - (timesteps - 1) * block_size
        if block_size > 0:
            indices = list(range(block_size//2, total_length - last_block_size, block_size))
        else:
            indices = []
        indices.append(block_size * (timesteps - 1) + last_block_size//2)

        return indices

    def _pad_to_max_length(self, data, labels, pad_by_last):
        data_lengths = torch.LongTensor([len(seq) for seq in data])

        max_length = data_lengths.max()
        data_tensor = torch.zeros((len(data_lengths), max_length, self.num_feature)).float()

        for i, length in enumerate(data_lengths):
            data_tensor[i, :length, :] = torch.FloatTensor(data[i])
            if pad_by_last:
                data_tensor[i, length:, :] = data_tensor[i, length-1, :]

        label_tensor = torch.LongTensor(labels)

        return data_tensor, label_tensor, data_lengths
# TODO class_name:['fall','stand','crouch','sit']
    def _get_label(self, data_dir):
        class_name = ['fall','stand']

        for idx in range(len(class_name)):
            if class_name[idx] in data_dir:
                return idx

    def _read_data(self, selected_points, root_dir=None):
        if root_dir is None:
            print('Data root directory not given.', file=sys.stderr)
            sys.exit(-1)

        data_X = []
        data_y = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.npy'):
                    d = np.load(os.path.join(root, file))

                    seq_length = len(d)
                    d = d[:, selected_points, :2]
                    d = d.reshape((seq_length, self.num_feature))

                    label = self._get_label(root)

                    data_X.append(d)
                    data_y.append(label)

        return data_X, data_y

