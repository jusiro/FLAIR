import torch
import numpy as np
import random
import os


"""
Seeds for reproducibility.
"""


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


"""
Download FLAIR pre-trained weights.
"""


def wget_gdrive_secure(fileid, filename):

    os.system("wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- | "
              "sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt".replace("$fileid", fileid))

    os.system("wget --load-cookies cookies.txt -O $filename"
              " 'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)"
              .replace("$fileid", fileid).replace("$filename", filename))