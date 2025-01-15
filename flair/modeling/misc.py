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


def wget_gdrive_secure(fileid, input_dir, filename):

    os.system("wget 'https://drive.usercontent.google.com/download?"
              "id=$fileid&"
              "export=download&"
              "authuser=0&"
              "confirm=t&"
              "uuid=40cc00ae-7d0b-4b86-b368-f0a37ebf480c&at=AIrpjvMO67CEnxRuJ6k2pvgHJSxq%3A1736964788644'"
              " -c -O '$filename'".
              replace("$fileid", fileid).replace("$filename", input_dir + filename))