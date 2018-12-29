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
    DBPL = ("../dbpl-dataset/", 22767)
    TOY = ("../toy_dataset/", 2268)
    DRUGS = ("../drugs-dataset/", 24)

def datasetchoice(choice):
    if choice == 'dbpl':
        return Dataset.DBPL
    if choice == 'toy':
        return Dataset.TOY
    if choice == 'drugs':
        return Dataset.DRUGS
    raise argparse.ArgumentTypeError('Expected dbpl or toy.')