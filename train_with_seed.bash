#!/bin/bash
#SBATCH -o seed_1.out

interact -q gpu -g 1 -t 24:00:00 -m 64g -f geforce3090

python /users/xhao9/Desktop/Lifelong_RL/mask_modified/train_continualworld.py ll_supermask --new_task_mask linear_comb --seed 456
