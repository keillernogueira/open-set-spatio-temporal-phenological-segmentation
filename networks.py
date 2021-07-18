import torch
import torch.nn.functional as F
from torchvision import models
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


class GRSL(nn.Module):
    def __init__(self, num_branches, input_channels, num_classes):  # 25x25
        super(GRSL, self).__init__()

        self.pools = torch.nn.ModuleList([self.conv_initial(input_channels) for _ in range(num_branches)])

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

    def conv_initial(self, input_channels):
        layer = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(3, stride=2)
        )
        return layer

    def forward(self, x):
        branches_out = torch.cat([cnn(inp) for inp, cnn in zip(x, self.pools)], dim=1)
        convs_out = self.convs(branches_out)
        convs_out = convs_out.view(convs_out.size(0), -1)  # flatten
        fc1_out = self.fc1(convs_out)
        fc2_out = self.fc2(fc1_out)
        final = self.final(fc2_out)
        return final, fc1_out, fc2_out


class FCN8s(nn.Module):
    def __init__(self, num_branches, input_channels, num_classes):
        super(FCN8s, self).__init__()

        # branches
        self.branches = torch.nn.ModuleList([self.conv_initial(input_channels) for _ in range(num_branches)])  # 12.5 = 13

        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*num_branches, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4  # 7.5 = 7
        )

        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8  # 3.5 = 4
        )

        # conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16 = 2
        )

        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32 = 1
        )

        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(256, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )

        self.score_fr = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(128, num_classes, 1)
        self.score_pool4 = nn.Conv2d(256, num_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 10, stride=5, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, bias=False)

        initialize_weights()

    def conv_initial(self, input_channels):
        layer = nn.Sequential(
            # conv1
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        )
        return layer

    def forward(self, x):
        branches_out = torch.cat([cnn(inp) for inp, cnn in zip(x, self.branches)], dim=1)

        conv2_out = self.conv2(branches_out)
        conv3_out = self.conv3(conv2_out)  # out
        conv4_out = self.conv4(conv3_out)   # out
        conv5_out = self.conv5(conv4_out)

        fc6_out = self.fc6(conv5_out)
        fc7_out = self.fc7(fc6_out)

        # upsample
        score1 = self.score_fr(fc7_out)
        upscore2 = self.upscore2(score1)

        score_pool4 = self.score_pool4(conv4_out)
        # h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        concat1 = upscore2 + score_pool4  # 1/16
        upscore_pool4 = self.upscore_pool4(concat1)

        score_pool3 = self.score_pool3(conv3_out)
        # h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        concat2 = upscore_pool4 + score_pool3  # 1/8
        out = self.upscore8(concat2)
        # h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return out, F.interpolate(conv5_out, x.size()[3:], mode='bilinear'), \
               F.interpolate(conv2_out, x.size()[3:], mode='bilinear')
