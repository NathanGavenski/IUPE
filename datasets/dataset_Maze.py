from collections import defaultdict
from copy import copy

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

np.set_printoptions(suppress=True)


def detect_path(file):
    return '/' in file


def read_data(file_path, dataset_path, reducted=False, no_hit=False):
    count = {}
    for i in range(-1, 4):
        count[str(i)] = []

    states = np.ndarray((0, 2))

    actions = []
    reducted_actions = []

    with open(file_path) as f:
        for idx, line in enumerate(f):
            words = line.replace('\n', '').split(';')
            state = words[0] if detect_path(words[0]) else f'{dataset_path}{words[0]}'
            nState = words[1] if detect_path(words[1]) else f'{dataset_path}{words[1]}'
            states = np.append(states, np.array([state, nState]).reshape((1, 2)), axis=0)

            action = int(words[-1])
            actions.append(action)

            if not no_hit:
                count[words[-1]].append(idx)

            reducted_action = 4 if int(words[-2]) == -1 else int(words[-1])
            reducted_actions.append(reducted_action)

            if no_hit:
                count[words[-2]].append(idx)

    return count, states, np.array(actions), np.array(reducted_actions)


def balance_dataset(file_path, dataset_path, downsample_size=4000, reducted=False, no_hit=False, replace=True, sampling=True):
    count, states, actions, reducted_actions = read_data(file_path, dataset_path, reducted, no_hit)

    sizes = []
    for key in count:
        sizes.append(len(count[key]))
    print(sizes)

    if sampling:
        max_size = np.min(sizes[1:]) if downsample_size is not None else None
        downsample_size = min(downsample_size, max_size)

    classes = list(range(0, 4))
    all_idxs = np.ndarray((0), dtype=np.int32)
    if downsample_size is not None:
        for i in classes:
            size = len(count[str(i)])

            try:
                random_idxs = np.random.choice(size, downsample_size, replace=replace)
            except ValueError:
                random_idxs = np.random.choice(size, size, replace=replace)

            idxs = np.array(count[str(i)])[random_idxs]
            all_idxs = np.append(all_idxs, idxs, axis=0).astype(int)

        states = states[all_idxs]
        if reducted:
            a = reducted_actions[all_idxs]
        else:
            a = actions[all_idxs]
    else:
        if reducted and no_hit:
            for i in classes:
                a = reducted_actions[count[str(i)]]
        else:
            for i in classes:
                a = actions

    print(np.bincount(a))
    return states, a


def split_dataset(states, actions, stratify=True):
    if stratify:
        return train_test_split(states, actions, test_size=0.3, stratify=actions)
    else:
        return train_test_split(states, actions, test_size=0.3)


class ExpertMaze():

    def __init__(self, runs, random, train):
        self.index = random[0, 0]
        self.train = train
        self.runs = runs
        self.original_runs = runs
        self.avg_reward = runs[:, 2].mean()
        self.avg_reward_random = random[:, 2].mean()

    def get_data(self, random, type):
        if type == 'all':
            split = int(len(self.runs) * self.train)
            return self.runs, self.runs

        if random is True:
            return self.get_random_data(type)
        else:
            return self.get_sample_data(type)

    def get_random_data(self, type):
        if type == 'same':
            index = np.random.randint(0, self.runs.shape[0])
            information = self.runs[index]
            self.runs = np.delete(self.runs, index, axis=0)
            return information, information
        else:
            result = []
            for _ in range(2):
                index = np.random.randint(0, self.runs.shape[0])
                result.append(self.runs[index])
                self.runs = np.delete(self.runs, index, axis=0)
            return result[0], result[1]

    def get_sample_data(self, type):
        if type == 'same':
            information = self.runs[0]
            self.runs = np.delete(self.runs, 0, axis=0)
            return information, information
        else:
            result = []
            result.append(self.runs[0])
            result.append(self.runs[1])
            self.runs = np.delete(self.runs, 0, axis=0)
            self.runs = np.delete(self.runs, 1, axis=0)
            return result[0], result[0]

    def reset(self):
        self.runs = copy(self.original_runs)


class IDM_Dataset(Dataset):

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def __init__(self, images, actions, mode='RGB'):
        super().__init__()
        self.previous_images = images[:, 0]
        self.next_images = images[:, 1]
        self.labels = actions
        self.mode = mode

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        s = Image.open(self.previous_images[idx]).convert(self.mode)
        nS = Image.open(self.next_images[idx]).convert(self.mode)

        s = self.transforms(s)
        nS = self.transforms(nS)

        a = torch.tensor(self.labels[idx])

        return (s, nS, a)


class Policy_Dataset(Dataset):

    transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def __init__(self, config, path, train=True,
                 random=True, type='single',
                 mode='RGB', labyrinths_valid=None,
                 train_size=0.7):
        super().__init__()

        self.mazes = []
        if type not in ['single', 'all', 'same', 'different']:
            raise Exception('Type should be \'all\', \'single\', \'same\' or \'different\'')

        self.read_expert(config + 'maze.txt')

        if labyrinths_valid is None:
            self.read_config(config, random, type, train_size)
        else:
            self.labyrinths_valid = labyrinths_valid

        if train:
            idxs = self.get_index(self.labyrinths_train)
            self.actions = self.actions[idxs.astype('int')]
            self.previous_images = self.previous_images[idxs.astype('int')]
            self.next_images = self.next_images[idxs.astype('int')]
        else:
            idxs = self.get_index(self.labyrinths_valid)
            self.actions = self.actions[idxs.astype('int')]
            self.previous_images = self.previous_images[idxs.astype('int')]
            self.next_images = self.next_images[idxs.astype('int')]

        self.mode = mode
        self.path = path

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        s = Image.open(f'{self.path}{self.previous_images[idx]}').convert(self.mode)
        nS = Image.open(f'{self.path}{self.next_images[idx]}').convert(self.mode)

        s = self.transforms(s)
        nS = self.transforms(nS)

        a = torch.tensor(self.actions[idx]).long()
        return (s, nS, a)

    def read_expert(self, file):
        self.actions = []
        self.next_images = []
        self.previous_images = []

        with open(file) as f:
            for line in f:
                words = line.replace('\n', '').split(';')
                self.previous_images.append(words[0])
                self.next_images.append(words[1])
                self.actions.append(words[-1])

        self.actions = np.array(self.actions, dtype=np.int32)
        self.next_images = np.array(self.next_images)
        self.previous_images = np.array(self.previous_images)

    def read_config(self, file, random, type, train):
        df = pd.read_csv(file + 'expert.txt', sep=';', header=None)
        df.columns = ['maze', 'number', 'reward', 'start', 'end']

        df_random = pd.read_csv(file + 'random.txt', sep=';', header=None)
        df_random.columns = ['maze', 'number', 'reward']

        for expert, random in zip(df.groupby('maze'), df_random.groupby('maze')):
            _, maze = expert
            _, random = random
            self.mazes.append(ExpertMaze(maze.to_numpy(), random.to_numpy(), train))

        self.labyrinths_train = []
        self.labyrinths_valid = []
        for labyrinth in self.mazes:
            train_lab, valid_lab = labyrinth.get_data(random, type)
            if type == 'all':
                for t in train_lab:
                    self.labyrinths_train.append(t)
                for v in valid_lab:
                    self.labyrinths_valid.append(v)
            else:
                self.labyrinths_train.append(train_lab)
                self.labyrinths_valid.append(valid_lab)

    def get_index(self, maze_list):
        idxs = np.array([])
        for i, labyrinth in enumerate(maze_list):
            r = labyrinth[-2:]
            begin = int(r[0])
            end = int(r[1]) + 1
            idxs = np.append(idxs, list(range(begin, end)))
        return idxs

    def get_performance_rewards(self, index):
        for maze in self.mazes:
            if maze.index == index:
                return maze.avg_reward, maze.avg_reward_random


def get_idm_dataset(
    path,
    batch_size,
    downsample_size=4000,
    reducted=False,
    shuffle=True,
    no_hit=False,
    replace=True,
    sampling=True
):
    file_path = f'{path}maze.txt'
    image_path = f'{path}images/'

    states, actions = balance_dataset(file_path, image_path, downsample_size, reducted, no_hit, replace, sampling)
    train, validation, train_y, validation_y = split_dataset(states, actions)

    train_dataset = IDM_Dataset(train, train_y)
    validation_dataset = IDM_Dataset(validation, validation_y)
    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation


def get_policy_dataset(
    path,
    batch_size,
    maze_size,
    maze_type,
    shuffle=Tru
):
    image_path = f'{path}maze{maze_size}/'
    config = f'{image_path}expert.txt'
    mazes = f'{image_path}maze.txt'
    expert = f'{image_path}random.txt'

    train_dataset = Policy_Dataset(image_path, image_path, random=False, type=maze_type)
    validation_dataset = Policy_Dataset(image_path, image_path, train=False, labyrinths_valid=train_dataset.labyrinths_valid)

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    validation = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle)
    return train, validation
