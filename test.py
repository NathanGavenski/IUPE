# Args should be imported before everything to cover https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
from utils.args import args
from collections import defaultdict
import os
import shutil

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from progress.bar import Bar
from models.idm import IDM
from models.idm import validation as validate_model
from datasets.dataset_Maze import IDM_Dataset
from datasets.dataset_Maze import get_policy_dataset

# ARGS: GPU and Pretrained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mazes_weights = os.listdir('./checkpoint/idm/')
for maze in mazes_weights:
    size = maze.replace('Maze', '')

    print('\nCreating PyTorch Datasets')
    print(f'Using dataset: {args.expert_path} with batch size: {args.batch_size}\n')
    train, validation = get_policy_dataset(args.expert_path, 16, args.maze_size, args.maze_type)

    data_kind = os.listdir(f'./checkpoint/idm/{maze}')
    for kind in data_kind:
        weights = os.listdir(f'./checkpoint/idm/{maze}/{kind}')
        print(f'Testing ./checkpoint/idm/{maze}/{kind}\n')

        best_weights = []
        for i, weight in enumerate(weights):
            bar = Bar(f'EPOCH {i}', max=len(validation), suffix='%(percent).1f%% - %(eta)ds')
            # w = os.listdir(f'./checkpoint/idm/{maze}/{kind}/{weight}')[0]
            w = f'./checkpoint/idm/{maze}/{kind}/{weight}'
            idm_model = IDM(args.actions, net=args.encoder, pretrained=args.pretrained)
            idm_model.load_state_dict(torch.load(w))
            idm_model = idm_model.to(device)
            idm_model.eval()

            epoch = []
            for itr, mini_batch in enumerate(validation):
                acc = validate_model(idm_model, mini_batch, device)
                epoch.append(acc)
                bar.next()

            print(f' - ACC: {round(np.mean(epoch), 2)} - {weight}')
            best_weights.append(np.mean(epoch))
        else:
            if len(best_weights) > 0:
                idx_best = np.argmax(best_weights, axis=0)
                src = f'./checkpoint/idm/{maze}/{kind}/{weights[idx_best]}/best_model.ckpt'
                dest = f'./best_weights/idm/{maze}/{kind}/'

                if os.path.exists(dest) is False:
                    os.makedirs(dest)

                print(f'\nCopying:\n\tsrc: {src}\n\tdest: {dest}\n')
                shutil.copy(src, dest)
