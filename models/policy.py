import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .general.mlp import MlpAttention
from .general.empty import Empty
from .general.resnet import create_resnet


class Policy(nn.Module):
    def __init__(self, action_size, net='vgg', pretrained=True, input=4):
        super(Policy, self).__init__()

        self.net = net

        if net == 'inception':
            self.model = models.inception_v3(pretrained=pretrained)
            self.model.fc = Empty()
            linear = nn.Linear(2048, 4096)
        elif net == 'vgg':
            self.model = models.vgg19_bn(pretrained=pretrained)
            self.model.classifier = Empty()
            self.fc_layers = nn.Sequential(
                nn.Linear((512 * 7 * 7), 4096),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, action_size)
            )

            if pretrained:
                print('Freezing weights')
                for params in self.model.parameters():
                    params.requires_grad = False

        elif net in ['resnet', 'attention-first', 'attention-last', 'attention-all']:
            self.model = create_resnet(net)(normalize=False)
            self.model.features.fc = Empty()
            self.fc_layers = nn.Sequential(
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, action_size)
            )

        elif net == 'vector':
            self.model = MlpAttention(input, action_size)

    def forward(self, state):
        if self.net == 'vector':
            x = self.model(state)
        else:
            s = self.model(state)
            x = self.fc_layers(s)
        return x


def train(model, idm_model, data, criterion, optimizer, device, tensorboard=None):
    if model.training is False:
        model.train()

    if idm_model.training is True:
        idm_model.eval()

    s, nS, a_gt = data

    if tensorboard is not None:
        tensorboard.add_grid(train=False, state=s, nState=nS)

    s = s.to(device)
    nS = nS.to(device)

    action = idm_model(s, nS)
    action = torch.argmax(action, 1)
    # action = a_gt.to(device)

    if tensorboard is not None:
        tensorboard.add_histogram('Valid/action distribution', action)
        tensorboard.add_histogram('Valid/GT action distribution', a_gt)

    optimizer.zero_grad()
    pred = model(s)

    pred_argmax = torch.argmax(pred, 1)
    loss = criterion(pred, action)
    loss.backward()
    optimizer.step()

    acc = ((pred_argmax == action).sum().item() / a_gt.shape[0]) * 100

    return loss, acc


def validation(model, idm_model, data, device, tensorboard=None):
    if model.training is True:
        model.eval()

    s, nS, a_gt = data

    if tensorboard is not None:
        tensorboard.add_grid(train=False, state=s, nState=nS)

    s = s.to(device)
    nS = nS.to(device)

    action = idm_model(s, nS)
    action = torch.argmax(action, 1)
    # action = a_gt.to(device)

    if tensorboard is not None:
        tensorboard.add_histogram('Valid/action distribution', action)
        tensorboard.add_histogram('Valid/GT action distribution', a_gt)

    pred = model(s)
    pred_argmax = torch.argmax(pred, 1)

    acc = ((pred_argmax == action).sum().item() / a_gt.shape[0]) * 100

    return acc
