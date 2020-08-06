import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .general.mlp import MlpAttention
from .general.empty import Empty
from .general.resnet import create_resnet


class IDM(nn.Module):

    def __init__(self, action_size, net='vgg', pretrained=True, input=8):
        super(IDM, self).__init__()

        self.net = net

        if net == 'inception':
            self.model = models.inception_v3(pretrained=pretrained)

        elif net == 'vgg':
            self.model = models.vgg19_bn(pretrained=pretrained)
            self.model.classifier = Empty()
            self.fc_layers = nn.Sequential(
                nn.Linear((512 * 7 * 7) * 2, 4096),
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
                nn.Linear(512 * 2, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, action_size)
            )

        elif net == 'vector':
            self.model = MlpAttention(input, action_size)

    def forward(self, state, nState):
        if self.net == 'vector':
            input = torch.cat((state, nState), 1)
            x = self.model(input)
        else:
            s = self.model(state)
            nS = self.model(nState)
            x = self.fc_layers(torch.cat((s, nS), 1))
        return x

    def get_params(self):
        return (x for x in torch.cat((self.reduction.parameters(), self.fc_layers.parameters())))


def train(model, data, criterion, optimizer, device, tensorboard=None):
    if model.training is False:
        model.train()

    s, nS, a = data

    if tensorboard is not None:
        tensorboard.add_grid(train=True, state=s, nState=nS)

    s = s.to(device)
    nS = nS.to(device)
    a = a.to(device)

    optimizer.zero_grad()
    pred = model(s, nS)

    pred_argmax = torch.argmax(pred, 1)
    loss = criterion(pred, a)
    loss.backward()
    optimizer.step()

    acc = ((pred_argmax == a).sum().item() / a.shape[0]) * 100
    return loss, acc


def validation(model, data, device, tensorboard=None):
    if model.training is True:
        model.eval()

    s, nS, a = data

    if tensorboard is not None:
        tensorboard.add_grid(train=False, state=s, nState=nS)

    s = s.to(device)
    nS = nS.to(device)
    a = a.to(device)

    pred = model(s, nS)

    pred_argmax = torch.argmax(pred, 1)

    acc = ((pred_argmax == a).sum().item() / a.shape[0]) * 100

    return acc
