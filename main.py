import glob
import unicodedata
import string
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn

import datetime
from abc import ABC, abstractmethod
import os
import copy
import json

from init_data import work_path, data_path, test_dict, test_param, trial_param, task_param
from rnn_cells import EncoderCell, DecoderCell, MyLinerRNN_cell, MyRNN_cell, MyLSTM_cell, MyGRU_cell
from pipeline import Trial, TestRNNArch
from tasks import *
# Preparing wark dirictory


if not os.path.exists(work_path):
    os.mkdir(work_path)

if not os.path.exists(data_path):
    os.mkdir(data_path)



RNNArch_list = [MyGRU_cell, MyLSTM_cell, MyRNN_cell]
testRNNArch = TestRNNArch()
testRNNArch.initialization(test_param, test_dict, trial_param, task_param, do_print=False)
sep = '\n'
out_str = testRNNArch.get_results_hat() + sep
# print(testRNNArch.get_results_hat())

out_dict = {}
i = 0
for RNNArch in RNNArch_list:
    testRNNArch.do_test(RNNArch, 1,0)
    out_dict[RNNArch.name] = testRNNArch.last_results
    str_ = testRNNArch.get_results()
    if i == 0:
        print(testRNNArch.get_results_hat())
        i =1
    print(str_)
    out_str += str_+ sep

t = datetime.datetime.now().strftime(trial_param['timemark_format'])

with open(os.path.join(data_path,'Rnn_exp_' + t + '.txt'), 'w') as file:
    file.write(out_str)

with open(os.path.join(data_path,'Rnn_exp_' + t + '.json'), 'w') as file:
    json.dump(out_dict, file, indent=4)


