
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
    "/media/backup/competition/train_models_swin_erasenet_finetune/STE_1_39.4660.pdparams",
    "/media/backup/competition/train_models_swin_erasenet_finetune/STE_5_39.7661.pdparams",
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
            avg_state_dict[k] = new_state_dict[k].clone()
            avg_counts[k] = 1
        else:
            if "position_index" in k:
                continue
            avg_state_dict[k] += v
            avg_counts[k] += 1

for k, v in avg_state_dict.items():
    if "position_index" in k:
        continue
    avg_state_dict[k]=avg_state_dict[k]/float(avg_counts[k])
new_state_dict1 = paddle.load(state_dicts[0])
new_state_dict2 = paddle.load(state_dicts[1])

print(new_state_dict1[k])
print(new_state_dict2[k])
print(avg_state_dict[k])

paddle.save(avg_state_dict, 'average_model.pdparams')