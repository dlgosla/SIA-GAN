import torch
import os
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

def seed_all(seed: int = 42):

    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # for numpy pseudo-random generator
    random.seed(seed)  # set fixed value for python built-in pseudo-random generator
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

