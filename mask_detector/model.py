from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, Sequential, Softmax, Flatten


class BasicCNN(Module):
    def __init__(self):
        super().__init__()

        self.convLayers1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayers2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayers3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.linearLayers = Sequential(
            Linear(in_features=128*25*25, out_features=2048),
            ReLU(),
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=1),
            Softmax()
        )

    def forward(self, x):
        x = self.convLayers1(x)
        x = self.convLayers2(x)
        x = self.convLayers3(x)
        x = x.view(-1, 128*25*25)
        x = self.linearLayers(x)

        return x
