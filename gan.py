import torch.nn as nn
import torch
import torch.nn.functional as F
import logging
import os

PID = os.getpid()
logger = logging.getLogger(str(PID))


class Generator(nn.Module):
    def __init__(self, c):
        # Padding mode can be 'zeros', 'reflect', 'replicate'
        super(Generator, self).__init__()

        input_size = c["g_input_size"] + c["c_dim"]
        conv_kernel = 5
        conv_pad = int(conv_kernel / 2)
        fake_data_output_size = c["seq_length"]

        self.map1_linear = nn.Linear(input_size, 150)
        self.map2_conv = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=conv_kernel,
            padding=conv_pad,
            padding_mode='replicate')
        self.map3_conv = nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=conv_kernel,
            padding=conv_pad,
            padding_mode='replicate')
        self.map4_conv = nn.Conv1d(
            in_channels=16,
            out_channels=8,
            kernel_size=conv_kernel,
            padding=conv_pad,
            padding_mode='replicate')
        self.map5_conv = nn.Conv1d(
            in_channels=8,
            out_channels=1,
            kernel_size=conv_kernel,
            padding=conv_pad)
        self.map6_linear = nn.Linear(
            in_features=150,
            out_features=125)
        self.map7_linear = nn.Linear(
            in_features=125,
            out_features=100)
        self.map8_linear = nn.Linear(
            in_features=100,
            out_features=fake_data_output_size)

    def forward(self, x):
        l_relu_slope = 0.2

        x = self.map1_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)
        x = x.unsqueeze(dim=1)

        x = self.map2_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map3_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map4_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map5_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = x.squeeze()

        x = self.map6_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map7_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map8_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        return x


class Discriminator(nn.Module):
    def __init__(self, c):
        super(Discriminator, self).__init__()

        num_convs = 32
        conv_kernel = 5
        conv_pad = int(conv_kernel / 2)
        map4_input = int((c["seq_length"] / 2 / 2) * num_convs)

        self.map1_conv = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=conv_kernel,
            padding=conv_pad,
            padding_mode='replicate')
        self.map2_conv = nn.Conv1d(
            in_channels=32,
            out_channels=16,
            kernel_size=conv_kernel,
            padding=conv_pad,
            padding_mode='replicate')
        self.map3_conv = nn.Conv1d(
            in_channels=16,
            out_channels=8,
            kernel_size=conv_kernel,
            padding=conv_pad,
            padding_mode='replicate')
        self.map4_linear = nn.Linear(
            in_features=192,
            out_features=50)
        self.map5_linear = nn.Linear(
            in_features=50,
            out_features=15)
        self.map6_linear = nn.Linear(
            in_features=15,
            out_features=1)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.pool3 = nn.MaxPool1d(kernel_size=1)

    def forward(self, x):
        l_relu_slope = 0.2

        x = x.unsqueeze(1)

        x = self.map1_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)
        x = self.pool1(x)

        x = self.map2_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)
        x = self.pool2(x)

        x = self.map3_conv(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.map4_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map5_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        x = self.map6_linear(x)
        x = F.leaky_relu(x, negative_slope=l_relu_slope)

        return x
