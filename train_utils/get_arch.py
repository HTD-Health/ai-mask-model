# Imports python modules
import torchvision.models


def get_arch(arch: str) -> torchvision.models:
    """
    Return object of the network architecture from torchvision library.

    Args:
        arch (str): name of architecture.

    Return:
        torchvision.models: object of the proper architecture class.
    """
    try:
        if arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
        elif arch == 'alexnet':
            model = torchvision.models.alexnet(pretrained=True)
        elif arch == 'vgg13':
            model = torchvision.models.vgg13(pretrained=True)
        elif arch == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
        else:
            raise ValueError(f"No matching architecture: {arch}")
    except ValueError as e:
        print(e)
    
    return model
