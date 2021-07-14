import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def conv_initial(input_channels):
    layer = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(inplace=True),

        nn.MaxPool2d(3, stride=2)
    )
    return layer


class GRSL(nn.Module):
    def __init__(self, num_branches, input_channels, num_classes):  # 25x25
        super(GRSL, self).__init__()

        self.pools = torch.nn.ModuleList([conv_initial(input_channels) for _ in range(num_branches)])

        self.convs = nn.Sequential(
            nn.Conv2d(num_branches*64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(3, stride=2),
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(3, stride=2)
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1*1*256, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.Linear(1024, num_classes)
        )

        initialize_weights(self)

    def forward(self, x):
        branches_out = torch.cat([cnn(inp) for inp, cnn in zip(x, self.pools)], dim=1)
        convs_out = self.convs(branches_out)
        convs_out = convs_out.view(convs_out.size(0), -1)  # flatten
        fc1_out = self.fc1(convs_out)
        fc2_out = self.fc2(fc1_out)
        final = self.final(fc2_out)
        return final, fc1_out, fc2_out
