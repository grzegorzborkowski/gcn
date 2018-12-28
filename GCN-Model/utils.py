import argparse
from enum import Enum

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Dataset(Enum):
    DBPL = "../dbpl-dataset/"
    TOY = "../toy_dataset/"

def datasetchoice(choice):
    if choice == 'dbpl':
        return Dataset.DBPL
    if choice == 'toy':
        return Dataset.TOY
    raise argparse.ArgumentTypeError('Expected dbpl or toy.')