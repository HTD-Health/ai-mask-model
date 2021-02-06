import torch
from torch.nn import AvgPool2d, Dropout, Flatten, Module, Conv2d, Linear, MaxPool2d, ReLU, Sequential, Sigmoid


class MobileNetV2(Module):
    def __init__(self):
        super().__init__()

        net = torch.hub.load('pytorch/vision:v0.6.0',
                             'mobilenet_v2', pretrained=True)

        self.features = net.features

        self.classifier = Sequential(
            AvgPool2d(kernel_size=(7, 7)),
            Flatten(),
            Linear(in_features=1280, out_features=128),
            Dropout(p=0.5),
            ReLU(),
            Linear(in_features=128, out_features=1),
            Sigmoid()
        )

    def forward(self, output):
        with torch.no_grad():
            output = self.features(output)

        output = self.classifier(output)

        return output
