import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image

#https://www.maxlist.xyz/2019/12/25/python-property/
class VideoRecord(object):

    def __init__(self, video_info):
        self._data = video_info

    @property                       #property that means class object can read
    def id(self):
        return self._data['id']

    @property
    def path(self):
        return self._data['path']

    @property
    def num_frames(self):
        return self._data['num_frames']

    @property
    def label(self):
        return self._data['label']


class TSNDataSet(data.Dataset):

    def __init__(self, info_list, data_root, num_segments, new_length=1,
                 modality='RGB', transform=None, random_sample=True):
        self.info_list = info_list
        self.data_root = data_root
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_sample = random_sample

        self._build_video_record_list()

        # Diff needs one more image to calculate difference
        if self.modality == 'RGBdiff':
            self.new_length += 1

    def _build_video_record_list(self):
        self.video_record_list = [VideoRecord(x) for x in self.info_list]

    def _random_sample_indices(self, num_frames):
        segment_size = (num_frames - self.new_length + 1) // self.num_segments

        offset = np.random.randint(segment_size, size=self.num_segments)
        offset += np.array([segment_size * x for x in range(self.num_segments)])

        return offset + 1

    def _middle_indices(self, num_frames):
        avg_frames = float(num_frames - self.new_length + 1) / self.num_segments

        offset = np.array([int(avg_frames / 2 + avg_frames * x) for x in range(self.num_segments)])

        return offset + 1

    def _read_images(self, record, indices):
        image_list = []
        record_id = record.id

        for i in indices:
            for p in range(self.new_length):
                image_path = os.path.join(self.data_root, '{}\\rgb\\rgb_{:04d}.jpg'.format(record_id, i+p))
                image_list.append(Image.open(image_path).convert('RGB'))

        return image_list

    def get_data_weight(self):
        cnt = [0] * 2
        labels = ['fall', 'stand']  # 'sit', 'crouch'

        for rec in self.video_record_list:
            idx = labels.index(rec.label)
            cnt[idx] += 1

        weight = [0] * len(self.video_record_list)
        for i, rec in enumerate(self.video_record_list):
            idx = labels.index(rec.label)
            weight[i] = 1 / cnt[idx]

        return weight

    def __getitem__(self, idx):
        video_record = self.video_record_list[idx]
        num_frames = video_record.num_frames

        if self.random_sample:
            indices = self._random_sample_indices(num_frames)
        else:
            indices = self._middle_indices(num_frames)

        image_list = self._read_images(video_record, indices)

        label = video_record.label
        if label == 'fall':
            label = 0
        elif label == 'stand':
            label = 1
        '''elif label == 'sit':
            label = 2
        elif label == 'crouch':
            label = 3'''

        if self.transform is not None:
            processed_data = self.transform(image_list)
            return processed_data, label
        else:
            return image_list, label

    def __len__(self):
        return len(self.video_record_list)


if __name__ == '__main__':
    dataSet = TSNDataSet(info_file_path='mod_code/openpose/code/extract/info.txt', data_root='extract',
                         num_segments=10, new_length=1, modality='RGB')

    print(len(dataSet))
