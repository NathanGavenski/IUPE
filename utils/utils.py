from collections import defaultdict
from copy import deepcopy
import os
from os import listdir
from os.path import isfile, join
import random
import shutil

import gym
import gym_maze
import numpy as np
import pandas as pd
from PIL import Image
import torch

from datasets.dataset_Maze import get_idm_dataset as maze_idm
from datasets.dataset_Maze import get_policy_dataset as maze_policy
from datasets.dataset_CartPole import get_idm_dataset as cartpole_idm
from datasets.dataset_CartPole import get_policy_dataset as cartpole_policy
from datasets.dataset_CartPole import get_idm_vector_dataset as cartpole_idm_vector
from datasets.dataset_CartPole import get_policy_vector_dataset as cartpole_policy_vector
from datasets.dataset_MountainCar import get_idm_dataset as mountaincar_idm
from datasets.dataset_MountainCar import get_policy_dataset as mountaincar_policy
from datasets.dataset_MountainCar import get_idm_vector_dataset as mountaincar_idm_vector
from datasets.dataset_MountainCar import get_policy_vector_dataset as mountaincar_policy_vector
from datasets.dataset_Acrobot import get_idm_dataset as acrobot_idm
from datasets.dataset_Acrobot import get_policy_dataset as acrobot_policy
from datasets.dataset_Acrobot import get_idm_vector_dataset as acrobot_idm_vector
from datasets.dataset_Acrobot import get_policy_vector_dataset as acrobot_policy_vector
from models.idm import IDM
from models.policy import Policy
from utils.enjoy import delete_alpha
from utils.enjoy import get_environment
from utils.enjoy import get_min_reward
from utils.enjoy import performance
from utils.enjoy import play
from utils.enjoy import play_vector
from utils.exceptions import CheckpointAlreadyExists


class CheckpointAlreadyExists(Exception):
    def __init__(self, files, name):
        super().__init__(f'{name} already exists in {files}')


def save_idm_model(model, replace=False, name=None, folder=None):
    parent_folder = './checkpoint/idm'
    path = folder if folder is not None else parent_folder
    if not os.path.exists(path):
        os.mkdir(path)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    checkpoint_name = f'model_{str(len(onlyfiles) + 1)}.ckpt' if name is None else name

    if checkpoint_name in onlyfiles and replace is False:
        raise CheckpointAlreadyExists(onlyfiles, checkpoint_name)
    elif replace is True:
        torch.save(model.state_dict(), f'{path}/best_model.ckpt')
    elif replace is False:
        torch.save(model.state.dict(), f'{path}/{checkpoint_name}')


def save_policy_model(model, replace=False, name=None, folder=None):
    parent_folder = './checkpoint/policy'
    path = folder if folder is not None else parent_folder
    if not os.path.exists(path):
        os.mkdir(path)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    checkpoint_name = f'model_{str(len(onlyfiles) + 1)}.ckpt' if name is None else name

    if checkpoint_name in onlyfiles and replace is False:
        raise CheckpointAlreadyExists(onlyfiles, checkpoint_name)
    elif replace:
        torch.save(model.state_dict(), f'{path}/best_model.ckpt')
    elif not replace:
        torch.save(model.state.dict(), f'{path}/{checkpoint_name}')


def load_policy_model(args, environment, device, folder=None):
    parent_folder = './checkpoint/policy'
    path = folder if folder is not None else parent_folder

    model = Policy(
        environment['action'],
        net=args.encoder,
        pretrained=args.pretrained,
        input=environment['input_size']
    )
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    model = model.to(device)
    model.eval()
    return model


def load_idm_model(args, folder=None):
    parent_folder = './checkpoint/policy'
    path = f'{parent_folder}/{folder}' if folder is not None else parent_folder

    model = IDM(args.actions, pretrained=args.pretrained)
    model.load_state_dict(torch.load(f'{path}/best_model.ckpt'))
    model = model.to(device)
    model.eval()
    return model


def save_gif(gif_images, random, iteration):
    if not os.path.exists('./gifs/'):
        os.makedirs('./gifs/')

    name = 'Random' if random else 'Sample'
    gif_images[0].save(f'./gifs/{name}_{iteration}.gif',
                       format='GIF',
                       append_images=gif_images[1:],
                       save_all=True,
                       duration=100,
                       loop=0)


def policy_infer(
    model,
    dataloader,
    device,
    domain,
    random=False,
    size=(10, 10),
    episodes=10,
    seed=None,
    bar=None,
    verbose=False,
    dataset=False,
    gif=True,
    alpha_location=None,
):

    _dataset = dataloader.dataset
    transforms = _dataset.transforms

    if model.training:
        model.eval()

    if dataset:
        delete_alpha(alpha_location, domain)

    count = 0
    total_solved = 0
    reward_epoch = []
    performance_epoch = []
    states = np.ndarray((0, 4), dtype=str)
    for e in range(episodes):
        _seed = e + 1 if seed is None else seed
        env = get_environment(domain, size, _seed, random)

        play_function = domain['enjoy']
        epoch_data = play_function(
            env,
            model,
            dataset,
            gif,
            count,
            alpha_location,
            transforms,
            device,
            states,
            domain,
        )
        total_reward, gif_images, count, goal = epoch_data
        total_solved += goal

        if gif:
            save_gif(gif_images, random, e)

        if not random:
            performance_epoch.append(performance(total_reward, _seed, _dataset))
        reward_epoch.append(total_reward)

        if verbose:
            print(f'{e}/{episodes} - Total Reward: {total_reward}')

        if bar is not None:
            bar.next()

        env.close()
        del env

    return np.mean(reward_epoch), np.mean(performance_epoch), total_solved / episodes


def create_alpha_dataset(idm, alpha, domain, ratio=None):
    try:
        alpha_file = open(f'{alpha}{domain["file"]}', 'r')

        classes_alpha = defaultdict(list)
        for line in alpha_file:
            words = line.replace('\n', '').split(';')
            classes_alpha[words[-1]].append(words)
        alpha_file.close()
    except FileNotFoundError:
        alpha_file = []
        classes_alpha = defaultdict(list)

    idm_file = open(f'{idm}{domain["file"]}', 'r')
    classes_idm = defaultdict(list)
    for line in idm_file:
        words = line.replace('\n', '').split(';')
        classes_idm[words[-1]].append(words)
    idm_file.close()

    new_dataset = defaultdict(list)
    idm_ratio = 1 if ratio is None else 1 - ratio
    for key in range(domain['action']):
        # Ipre
        k = max(0, int(len(classes_idm[str(key)]) * idm_ratio))
        new_dataset[str(key)] += random.sample(classes_idm[str(key)], k)

        # Ipos
        k = max(0, int(len(classes_alpha[str(key)]) * ratio))
        new_dataset[str(key)] += random.sample(classes_alpha[str(key)], k)

    if not os.path.exists(alpha):
        os.makedirs(alpha)

    with open(f'{alpha}{domain["file"]}', 'w') as f:
        for key in new_dataset:
            for row in new_dataset[str(key)]:
                if 'previous' in row[0] and 'maze' in domain['name']:
                    f.write(f'{idm}images/{row[0]};{idm}images/{row[1]};{row[-2]};{row[-1]}\n')
                elif 'prev' in row[0]:
                    f.write(f'{idm}{row[0]};{idm}{row[1]};{row[-1]};{row[-1]}\n')
                else:
                    f.write(f'{alpha}images/{row[0]};{alpha}images/{row[1]};{row[-1]};{row[-1]}\n')


def create_alpha_vector_dataset(idm, alpha, domain, ratio=None):
    try:
        alpha_file = open(f'{alpha}{domain["file"]}', 'r')

        classes_alpha = defaultdict(list)
        for line in alpha_file:
            words = line.replace('\n', '').split(';')
            classes_alpha[words[-1]].append(line)
        alpha_file.close()
    except FileNotFoundError:
        alpha_file = []
        classes_alpha = defaultdict(list)

    idm_file = open(f'{idm}{domain["file"]}', 'r')
    classes_idm = defaultdict(list)
    for line in idm_file:
        words = line.replace('\n', '').split(';')
        classes_idm[words[-1]].append(line)
    idm_file.close()

    # ratio = 1 if ratio is None else 1 - ratio
    for key in range(domain['action']):
        if domain['name'] == 'maze':
            k = int(len(classes_idm[str(key)]) * ratio)
        else:
            if ratio == 0:
                k = int(len(classes_idm[str(key)]))
            else:
                action_size = (len(classes_alpha[str(key)]) * 100) / (ratio * 100)
                k = int(action_size - len(classes_alpha[str(key)]))

        size = len(classes_alpha[str(key)])
        classes_alpha[str(key)] += list(np.random.choice(classes_idm[str(key)], k))

        print(key, ratio, k, size, len(classes_alpha[str(key)]))

    if not os.path.exists(alpha):
        os.makedirs(alpha)

    with open(f'{alpha}{domain["file"]}', 'w') as f:
        for key in classes_alpha:
            f.write(''.join(classes_alpha[str(key)]))


domain = {
    'maze': {
        'file': 'maze.txt',
        'name': 'maze',
        'idm_dataset': maze_idm,
        'policy_dataset': maze_policy,
        'action': 4,
        'enjoy': play,
        'alpha': create_alpha_dataset,
        'input_size': None,
    },
    'cartpole_vector': {
        'file': 'cartpole.txt',
        'name': 'cartpole',
        'idm_dataset': cartpole_idm_vector,
        'policy_dataset': cartpole_policy_vector,
        'action': 2,
        'enjoy': play_vector,
        'alpha': create_alpha_vector_dataset,
        'input_size': 4,
    },
    'mountaincar_vector': {
        'file': 'mountaincar.txt',
        'name': 'mountaincar',
        'idm_dataset': mountaincar_idm_vector,
        'policy_dataset': mountaincar_policy_vector,
        'action': 3,
        'enjoy': play_vector,
        'alpha': create_alpha_vector_dataset,
        'input_size': 2,
    },
    'acrobot_vector': {
        'file': 'acrobot.txt',
        'name': 'acrobot',
        'idm_dataset': acrobot_idm_vector,
        'policy_dataset': acrobot_policy_vector,
        'action': 3,
        'enjoy': play_vector,
        'alpha': create_alpha_vector_dataset,
        'input_size': 6,
    }
}
