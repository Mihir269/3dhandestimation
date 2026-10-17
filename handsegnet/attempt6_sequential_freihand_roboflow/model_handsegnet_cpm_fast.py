import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),

            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

    def forward(self, x):
        return self.features(x)


class CPMStage(nn.Module):

    def __init__(self, in_ch):
        super().__init__()

        self.stage = nn.Sequential(

            ConvBlock(in_ch, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),

            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.stage(x)


class HandSegNetCPMFast(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = FeatureExtractor()

        self.stage1 = CPMStage(128)

        self.stage2 = CPMStage(129)

        self.stage3 = CPMStage(129)

    def forward(self, x):

        features = self.features(x)

        s1 = self.stage1(features)

        s1_prob = torch.sigmoid(s1)

        s2_input = torch.cat([features, s1_prob], dim=1)
        s2 = self.stage2(s2_input)

        s2_prob = torch.sigmoid(s2)

        s3_input = torch.cat([features, s2_prob], dim=1)
        s3 = self.stage3(s3_input)

        return s1, s2, s3
