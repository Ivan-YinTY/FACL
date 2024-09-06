import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision


class NormLinear(nn.Module):
    def __init__(self, input, output):
        super(NormLinear, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.reset_parameters()

    def forward(self, input):
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        input_normalized = F.normalize(input, p=2, dim=1)
        # print('W & I', weight_normalized.shape, input_normalized.shape)
        output = input_normalized.matmul(weight_normalized.t())
        return output

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, fea_size=128, nb_class=318, input_size=(1, 64, 128), loss='softmax'):
        super(ConvNet, self).__init__()
        self.loss = loss
        kernel_size = 5
        padding_size = 2 if kernel_size == 5 else 1
        self.conv1_1 = nn.Conv2d(input_size[0], 32, kernel_size, stride=1, padding=padding_size)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size, stride=1, padding=padding_size)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size, stride=1, padding=padding_size)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size, stride=1, padding=padding_size)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size, stride=1, padding=padding_size)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size, stride=1, padding=padding_size)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * (input_size[1]//8) * (input_size[2]//8), fea_size)
        self.prelu_fc1 = nn.PReLU()

        if self.loss == 'softmax':
            self.fc2 = nn.Linear(fea_size, nb_class, bias=False)
        elif self.loss == 'cosface' or self.loss == 'tripletCosface':
            self.fc2 = NormLinear(fea_size, nb_class)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.prelu_fc1(self.fc1(x))
        if self.loss == 'triplet':
            return x
        y = self.fc2(x)
        return x, y


class ResNet18(torch.nn.Module):
    def __init__(self, loss, num_classes):
        super(ResNet18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])

        if loss == 'softmax' or loss == 'centerloss' or loss == 'tripletSoftmax':
            self.fc = nn.Linear(512, num_classes, bias=False)
        elif loss == 'cosface' or loss == 'tripletCosface' or loss == 'tripletArcface' or loss == 'arcface':
            self.fc = NormLinear(512, num_classes)

    def forward(self, x):
        h = torch.squeeze(self.encoder(x))
        return h, self.fc(h)
