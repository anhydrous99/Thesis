from torch import Tensor
import torch.nn.functional as func
import numpy as np


a = Tensor(np.array([[[[1, 0, 2, 3], [4, 6, 6, 8], [3, 1, 1, 0], [1, 2, 2, 4]]]]))
w = Tensor(np.array([[[[1, 2], [3, 4]]]]))

o = func.conv2d(a, w, stride=1)
print(o)