import numpy as np
from pathlib import Path
import torch





if __name__ == '__main__':
    path = Path("data/fall/0000.npy")
    var = np.load(path)
    var = torch.from_numpy(var)
    print(var.shape)