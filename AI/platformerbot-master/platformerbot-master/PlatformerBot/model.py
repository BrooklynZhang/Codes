import torch
import numpy as np
import math
import PIL
from torch.nn.functional import leaky_relu


class CommonCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (320, 240)

        self.conv1 = torch.nn.Conv2d(3, 32, (8, 8), stride=4, padding=0)
        self.conv1_size = ((self.input_size[0] - 7 - 1) // 4 + 1, (self.input_size[1] - 7 - 1) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(32, 64, (4, 4), stride=2, padding=0)
        self.conv2_size = ((self.conv1_size[0] - 3 - 1) // 2 + 1, (self.conv1_size[1] - 3 - 1) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(64, 64, (3, 3), stride=1, padding=0)
        self.conv3_size = ((self.conv2_size[0] - 2 - 1) // 1 + 1, (self.conv2_size[1] - 2 - 1) // 1 + 1)

        #self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.pool4_size = ((self.conv3_size[0] - 1 - 1) // 2 + 1, (self.conv3_size[1] - 1 - 1) // 2 + 1)
        self.output_size = self.conv3_size
        print(self.output_size)

    def forward(self, img):
        img = torch.nn.functional.interpolate(img, self.input_size)
        yconv1 = self.conv1(img).relu()
        yconv2 = self.conv2(yconv1).relu()
        yconv3 = self.conv3(yconv2).relu()
        #ypool4 = self.maxpool4(yconv3)

        return yconv3

    def forward_detailed(self, img):
        img = torch.nn.functional.interpolate(img, self.input_size)
        yconv1 = leaky_relu(self.conv1(img))
        yconv2 = leaky_relu(self.conv2(yconv1))
        yconv3 = leaky_relu(self.conv3(yconv2))

        return {
            "img": img,
            "yconv1": yconv1,
            "yconv2": yconv2,
            "yconv3": yconv3,
        }


class PolicyNetwork(torch.nn.Module):
    def __init__(self, common_cnn=None):
        super().__init__()
        self.input_size = (320, 240)
        self.output_size = 22

        self.conv = common_cnn
        self.conv_size = common_cnn.output_size

        self.fc1_size = 512
        self.fc1 = torch.nn.Linear(64 * self.conv_size[0] * self.conv_size[1], self.fc1_size)
        self.fc2_size = 128
        self.fc2 = torch.nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3_size = self.output_size
        self.fc3 = torch.nn.Linear(self.fc2_size, self.fc3_size)

    def forward(self, img):
        yconv = self.conv(img)
        flatten = yconv.view(-1, 64 * self.conv_size[0] * self.conv_size[1])

        vfc1 = self.fc1(flatten).relu()
        vfc2 = self.fc2(vfc1).relu()
        vfc3 = self.fc3(vfc2).sigmoid()
        return vfc3

    def forward_detailed(self, img):
        conv_info = self.conv.forward_detailed(img)
        yconv = conv_info["yconv3"]
        flatten = yconv.view(-1, 64 * self.conv_size[0] * self.conv_size[1])

        vfc1 = self.fc1(flatten).relu()
        vfc2 = self.fc2(vfc1).relu()
        vfc3 = self.fc3(vfc2).sigmoid()
        return {
            "common-cnn": conv_info,
            "policy": {
                "flatten": flatten,
                "vfc1": vfc1,
                "vfc2": vfc2,
                "vfc3": vfc3,
            },
        }


class ValueNetwork(torch.nn.Module):
    def __init__(self, common_cnn=None):
        super().__init__()
        self.input_size = (320, 240)
        self.output_size = 1

        self.conv = common_cnn
        self.conv_size = common_cnn.output_size

        self.fc1_size = 512
        self.fc1 = torch.nn.Linear(64 * self.conv_size[0] * self.conv_size[1], self.fc1_size)
        self.fc2_size = 128
        self.fc2 = torch.nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3_size = 1
        self.fc3 = torch.nn.Linear(self.fc2_size, self.fc3_size)

    def forward(self, img):
        yconv = self.conv(img)
        flatten = yconv.view(-1, 64 * self.conv_size[0] * self.conv_size[1])

        vfc1 = self.fc1(flatten).relu()
        vfc2 = self.fc2(vfc1).relu()
        vfc3 = self.fc3(vfc2)

        return vfc3

    def forward_detailed(self, img):
        conv_info = self.conv.forward_detailed(img)
        yconv = conv_info["yconv3"]
        flatten = yconv.view(-1, 64 * self.conv_size[0] * self.conv_size[1])

        vfc1 = self.fc1(flatten).relu()
        vfc2 = self.fc2(vfc1).relu()
        vfc3 = self.fc3(vfc2)
        return {
            "common-cnn": conv_info,
            "value": {
                "flatten": flatten,
                "vfc1": vfc1,
                "vfc2": vfc2,
                "vfc3": vfc3,
            },
        }


if __name__ == "__main__":
    # for testing
    model = PolicyNetwork()
    zero_img = np.zeros((1, 3, 320, 240))
    out = model(torch.tensor(zero_img, dtype=torch.float))
    result = out.detach().cpu().numpy()
    print(result)
    print(result.shape)
