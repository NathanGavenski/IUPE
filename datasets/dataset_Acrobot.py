from collections import defaultdict
from copy import copy

import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

np.set_printoptions(suppress=True)


def detect_path(file):
    return '/' in file


def read_vector(dataset_path, idm=True):
    count = defaultdict(list)
    actions = []
    states = np.ndarray((0, 2, 6), dtype=str)
    with open(f'{dataset_path}acrobot.txt') as f:
        for idx, line in enumerate(f):
            word = line.replace('\n', '').split(';')
            state = np.fromstring(word[0].replace('[', '').replace(']', ''), sep=',', dtype=float)
            nState = np.fromstring(word[1].replace('[', '').replace(']', ''), sep=',', dtype=float)

            s = np.append(state[None], nState[None], axis=0)

            action = int(word[-1])
            actions.append(action)

            states = np.append(states, s[None], axis=0)

            count[action].append(idx)

    return count, states, np.array(actions)


def balance_dataset(dataset_path, downsample_size=5000, replace=True, sampling=True):
    data = read_vector(dataset_path)
    count, states, actions = data

    sizes = []
    dict_sizes = {}
    for key in count:
        sizes.append(len(count[key]))
        dict_sizes[key] = len(count[key])
    print('Size each action:', dict_sizes)

    if sampling:
        max_size = np.min(sizes) if downsample_size is not None else None
        downsample_size = min(downsample_size, max_size)

    classes = list(range(0, 3))
    all_idxs = np.ndarray((0), dtype=np.int32)
    if downsample_size is not None:
        for i in classes:
            size = len(count[i])

            try:
                random_idxs = np.random.choice(size, downsample_size, replace=replace)
            except ValueError:
                random_idxs = np.random.choice(size, size, replace=replace)

            idxs = np.array(count[i])[random_idxs]
            all_idxs = np.append(all_idxs, idxs, axis=0).astype(int)

        states = states[all_idxs]
        a = actions[all_idxs]
    else:
        for i in classes:
            a = actions

    print('Final size action:', np.bincount(a))
    return states, a


def split_dataset(states, actions, stratify=True):
    if stratify:
        return train_test_split(states, actions, test_size=0.3, stratify=actions)
    else:
        return train_test_split(states, actions, test_size=0.3)


class IDM_Vector_Dataset(Dataset):

    transforms = torch.from_numpy

    def __init__(self, path, images, actions):
        super().__init__()
        self.previous_images = images[:, 0]
        self.next_images = images[:, 1]
        self.labels = actions

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = self.previous_images[idx].astype(float)
        nS = self.next_images[idx].astype(float)

        s = torch.from_numpy(s)
        nS = torch.from_numpy(nS)

        a = torch.tensor(self.labels[idx])

        return (s, nS, a)


class Policy_Vector_Dataset(Dataset):

    transforms = torch.from_numpy

    random_policy = -482.6
    expert = -90.87272727272727

    def __init__(self, path, images, actions):
        super().__init__()
        self.previous_images = images[:, 0]
        self.next_images = images[:, 1]
        self.labels = actions

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = self.previous_images[idx].astype(float)
        nS = self.next_images[idx].astype(float)

        s = torch.from_numpy(s)
        nS = torch.from_numpy(nS)

        a = torch.tensor(self.labels[idx])

        return (s, nS, a)

    def get_performance_rewards(self, *args):
        return self.expert, self.random_policy


def get_idm_vector_dataset(
    path,
    batch_size,
    downsample_size=5000,
    shuffle=True,
    replace=True,
    sampling=True,
    **kwargs
):

    states, actions = balance_dataset(
        path,
        downsample_size=downsample_size,
        replace=replace,
        sampling=sampling,
        vector=True,
    )
    train, validation, train_y, validation_y = split_dataset(states, actions)

    train_dataset = IDM_Vector_Dataset(path, train, train_y)
    validation_dataset = IDM_Vector_Dataset(path, validation, validation_y)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation


def get_policy_vector_dataset(
    path,
    batch_size,
    downsample_size=5000,
    shuffle=True,
    replace=True,
    sampling=True,
    **kwargs
):

    states, actions = balance_dataset(
        path,
        downsample_size=None,
        replace=False,
        sampling=False,
        vector=True,
    )
    train, validation, train_y, validation_y = split_dataset(states, actions)

    train_dataset = Policy_Vector_Dataset(path, train, train_y)
    validation_dataset = Policy_Vector_Dataset(path, validation, validation_y)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation
