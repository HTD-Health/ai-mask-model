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

        self.linearLayers = Sequential(
            Linear(in_features=64*8*8, out_features=1024),
            Softmax(),
            Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        print(x.size())
        x = self.convLayers1(x)
        print(x.shape)
        x = self.convLayers2(x)
        print(x.shape)
        x = x.view(-1, 64*8*8)
        print(x.shape)
        x = self.linearLayers(x)
        print(x.shape)

        return x
