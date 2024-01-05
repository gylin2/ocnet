import torch
import os
import random
import numpy as np

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

def L2_regularization(model, L2_alpha):
    L2_loss = 0
    for param in model.parameters():
        L2_loss += torch.sum(param**2)/2.0
    return L2_alpha* L2_loss
