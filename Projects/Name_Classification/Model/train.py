from __future__ import unicode_literals, print_function, division

import os
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import re
from torchvision import datasets, models, transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch 
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

from utils import *
from model import RNN
import time
import math
import unicodedata
import string
from io import open
import glob
import os


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

category_lines = {}
all_categories = []

for filename in findFiles('../data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

criterion = nn.NLLLoss()

def train(rnn, category_tensor, line_tensor, learning_rate = 0.005):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--lr", help="learning rate",default=0.005, type=float)
parser.add_argument("--hidden", help="hidden states", default = 128, type=int)
parser.add_argument("--n_iters", help="number of iterations", default= 100000,type=int)
parser.add_argument("--output", help="path where you want to save the resulting network",type=str)


if(__name__ == "__main__"):

    writer = SummaryWriter()

    args = parser.parse_args()

    n_iters = args.n_iters
    print_every = n_iters // 10
    plot_every = n_iters // 10000

    n_hidden = args.hidden
    learning_rate = args.lr


    current_loss = 0

    rnn = RNN(n_letters, n_hidden, n_categories)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(rnn ,category_tensor, line_tensor, learning_rate)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:


            if(args.output != None):
                torch.save(rnn.state_dict(), "./" + str(args.output))

            writer.add_scalar('Loss',current_loss / plot_every , iter)
            current_loss = 0




