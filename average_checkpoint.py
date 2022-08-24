
from collections import defaultdict
from pyexpat import model
import sys
from matplotlib import lines
import os
sys.path.append('.')
import paddle

avg_state_dict = {}
avg_counts = {}

state_dicts = [
    "train_models_swin_erasenet_finetune/STE_1_37.9850.pdparams",
    "train_models_swin_erasenet_finetune/STE_3_38.0509.pdparams",
    "train_models_swin_erasenet_finetune/STE_7_38.0395.pdparams",
    "train_models_swin_erasenet_finetune/STE_9_38.2583.pdparams",
    "train_models_swin_erasenet_finetune/STE_8_38.1507.pdparams",
    "train_models_swin_erasenet_finetune/STE_11_38.2465.pdparams",
    "train_models_swin_erasenet_finetune/STE_12_38.2562.pdparams",
    "train_models_swin_erasenet_finetune/STE_13_38.2906.pdparams",
         ]


avg_state_dict = {}
avg_counts = {}
for c in state_dicts:
    new_state_dict = paddle.load(c)
    if not new_state_dict:
        print("Error: Checkpoint ({}) doesn't exist".format(c))
        continue
    for k, v in new_state_dict.items():
        if k not in avg_state_dict:
            avg_state_dict[k] = v.clone()
            avg_counts[k] = 1
        else:
            if "position_index" in k:
                continue
            avg_state_dict[k] += v
            avg_counts[k] += 1

for k, v in avg_state_dict.items():
    if "position_index" in k:
        continue
    avg_state_dict[k]=v/avg_counts[k]

paddle.save(avg_state_dict, 'average_model.pdparams')