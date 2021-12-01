import numpy as np
from pathlib import Path
import torch





if __name__ == '__main__':
    path = Path("data/fall/0000.npy")
    var = np.load(path)
    var = torch.from_numpy(var)
    num_keypoint = 25
    feature_array = np.zeros([45, num_keypoint, 3], np.float32)
    print(feature_array.shape)
    print(len(feature_array))