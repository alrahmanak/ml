# python predict.py /path/to/image checkpoint

import sys
import argparse
import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn
from torch import optim
import torch.nn.functional as F
import collections
from collections import OrderedDict
import json
import time

print ('Predict Image', time.time())
print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

parser = argparse.ArgumentParser(description='Image classifier training')