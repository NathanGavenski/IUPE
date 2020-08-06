# Args should be imported before everything to cover https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
from utils.args import args
import numpy as np
import os
import sys
import torch
import torchvision

from PIL import Image
from progress.bar import Bar
from tensorboard_wrapper.tensorboard import Tensorboard as Board
from torch import nn, optim

from models.idm import IDM
from models.idm import train as train_idm
from models.idm import validation as validate_idm
from models.policy import Policy
from models.policy import train as train_policy
from models.policy import validation as validate_policy
from utils.utils import create_alpha_dataset
from utils.utils import domain
from utils.utils import policy_infer
from utils.utils import save_policy_model


# ARGS: GPU and Pretrained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment = domain[args.domain]

# Tensorboard
parent_folder = "_".join(args.run_name.split('_')[:-1])
st_char = environment['name'][0].upper()
rest_char = environment['name'][1:]
if 'maze' in environment['name']:
    env_name = f'{st_char}{rest_char}{args.maze_size}'
else:
    env_name = f'{st_char}{rest_char}'

name = f'./checkpoint/alpha/{env_name}/{parent_folder}/{args.run_name}'
if os.path.exists(name) is False:
    os.makedirs(name)

parent_folder = "_".join(args.run_name.split('_')[:-1])
path = f'./runs/alpha/{env_name}/{parent_folder}/{args.run_name}'
if os.path.exists(path) is False:
    os.makedirs(path)

board = Board(name, path)

# Datasets
print('\nCreating PyTorch IDM Datasets')
print(f'Using dataset: {args.data_path} with batch size: {args.batch_size}')
get_idm_dataset = environment['idm_dataset']
idm_train, idm_validation = get_idm_dataset(
    args.data_path,
    args.batch_size,
    downsample_size=5000,
    reducted=args.reducted,
    no_hit=args.no_hit
)


print('\nCreating PyTorch Policy Datasets')
print(f'Using dataset: {args.expert_path} with batch size: {args.policy_batch_size}')
get_policy_dataset = environment['policy_dataset']
policy_train, policy_validation = get_policy_dataset(
    args.expert_path,
    args.policy_batch_size,
    maze_size=args.maze_size,
    maze_type=args.maze_type,
)

# Model and action size
print('\nCreating Models')
action_dimension = environment['action']
inputs = environment['input_size'] * 2 if environment['input_size'] is not None else None
policy_model = Policy(
    action_dimension, 
    net=args.encoder, 
    pretrained=args.pretrained, 
    input=environment['input_size']
)
idm_model = IDM(
    action_dimension,
    net=args.encoder,
    pretrained=args.pretrained,
    input=inputs
)

policy_model.to(device)
idm_model.to(device)

# Optimizer and loss
print('\nCreating Optimizer and Loss')
print(f'IDM learning rate: {args.lr}\nPolicy learning rate: {args.policy_lr}')
idm_lr = args.lr
idm_criterion = nn.CrossEntropyLoss()
idm_optimizer = optim.Adam(idm_model.parameters(), lr=idm_lr)

policy_lr = args.policy_lr
policy_criterion = nn.CrossEntropyLoss()
policy_optimizer = optim.Adam(policy_model.parameters(), lr=policy_lr)

# Learning rate decay
print('Setting up Learning Rate Decay function and Schedulers')
idm_lr_decay = lambda epoch: args.lr_decay_rate**epoch
policy_lr_decay = lambda epoch: args.policy_lr_decay_rate**epoch

idm_scheduler = optim.lr_scheduler.LambdaLR(idm_optimizer, idm_lr_decay)
policy_scheduler = optim.lr_scheduler.LambdaLR(policy_optimizer, policy_lr_decay)

# Train
print('Starting Train\n')
best_epoch_acc = 0
early_stop_count = 0
max_epochs = args.idm_epochs
max_iter = len(idm_train) + len(idm_validation) + len(policy_train) + len(policy_validation)

for epoch in range(max_epochs):

    board.add_scalar('Learning Rate', idm_lr_decay(epoch - 1))

    ############################ IDM Train ############################
    if args.verbose is True:
        bar = Bar(f'EPOCH {epoch:3d}', max=max_iter, suffix='%(percent).1f%% - %(eta)ds')

    batch_acc = []
    batch_loss = []
    for itr, mini_batch in enumerate(idm_train):
        loss, acc = train_idm(
            idm_model,
            mini_batch,
            idm_criterion,
            idm_optimizer,
            device,
            board,
        )

        batch_acc.append(acc)
        batch_loss.append(loss.item())

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    ############################ IDM Validation ############################
    board.add_scalars(
        train=True,
        IDM_Loss=np.mean(batch_loss),
        IDM_Accuracy=np.mean(batch_acc)
    )

    batch_acc = []
    for itr, sample_batched in enumerate(idm_validation):
        acc = validate_idm(
            idm_model,
            mini_batch,
            device,
            board,
        )
        batch_acc.append(acc)

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    board.add_scalars(
        train=False,
        IDM_Accuracy=np.mean(batch_acc)
    )

    ############################ Policy Train ############################
    batch_acc = []
    batch_loss = []
    for itr, mini_batch in enumerate(policy_train):
        loss, acc = train_policy(
            policy_model,
            idm_model,
            mini_batch,
            policy_criterion,
            policy_optimizer,
            device,
            board
        )

        batch_acc.append(acc)
        batch_loss.append(loss.item())

        if args.verbose is True:
            bar.next()

        if args.debug:
            break

    board.add_scalars(
        train=True,
        Policy_Loss=np.mean(batch_loss),
        Policy_Accuracy=np.mean(batch_acc)
    )

    ############################ Policy Validation ############################
    batch_acc = []
    for itr, mini_batch in enumerate(policy_validation):
        acc = validate_policy(
            policy_model,
            idm_model,
            mini_batch,
            device,
            board
        )

        batch_acc.append(acc)
        bar.next()

    board.add_scalars(
        train=False,
        Policy_Accuracy=np.mean(batch_acc)
    )

    ############################ Policy Eval ############################
    if args.verbose is True:
        bar = Bar(
            f'VALID Sample {epoch:3d}',
            max=100,
            suffix='%(percent).1f%% - %(eta)ds'
        )
    else:
        bar = None

    if args.debug:
        amount = 1
    else:
        amount = 100

    infer, performance, solved = policy_infer(
        policy_model,
        dataloader=policy_train,
        device=device,
        domain=environment,
        size=(args.maze_size, args.maze_size),
        bar=bar,
        episodes=amount,
        gif=False,
        alpha_location=args.alpha,
        dataset=True,
    )

    board.add_scalars(
        train=False,
        AER_Sample=infer,
        Sample_Solved=solved,
        Performance_Sample=performance
    )

    if args.verbose is True:
        bar = Bar(
            f'VALID Random {epoch:3d}',
            max=10,
            suffix='%(percent).1f%% - %(eta)ds'
        )
    else:
        bar = None

    if args.debug:
        amount = 1
    else:
        amount = 10

    infer_random, _, random_solved = policy_infer(
        policy_model,
        dataloader=policy_train,
        device=device,
        domain=environment,
        random=True,
        size=(args.maze_size, args.maze_size),
        episodes=amount,
        bar=bar,
        gif=False,
    )

    board.add_scalars(
        train=False,
        AER_Random=infer_random,
        Random_Solved=random_solved
    )
    print(f'\nSample Infer {infer}\tRandom Infer {infer_random}\n')

    print(f'Using dataset: {args.alpha} with batch size: {args.batch_size}')

    create_alpha = environment['alpha']
    create_alpha(
        args.data_path,
        args.alpha,
        environment,
        solved,
    )

    idm_train, idm_validation = get_idm_dataset(
        args.alpha,
        args.batch_size,
        downsample_size=np.inf,
        reducted=args.reducted,
        replace=False,
        sampling=False,
        no_hit=args.no_hit,
    )

    ############################ Necessary updates ############################
    board.step()
