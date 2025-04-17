import logging
import os
import tarfile

import torch
from torch.nn.utils.rnn import pad_sequence
from process_util import split_dataset 
data = {
    'feature': [1, 2, 3, 4, 5],
    'label': [0, 1, 0, 1, 0]
}
split_idxs = {
    'train': [0, 1, 2],
    'test': [3, 4]
}
data_split = split_dataset(data, split_idxs) 