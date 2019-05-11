# import matplotlib.pyplot as plt
import os, sys, torch
from model import resnet6
from model_resnet import ResNet50


def show_image(image):
    dpi = 80
    figsize = (image.shape[1] / float(dpi), image.shape[0] / float(dpi))
    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    fig.show()


def get_file_name(filepath):
    return os.path.basename(filepath).split(".")[0]


def get_model(arch, num_classes, pretrained_model):
    if arch == "resnet6":
        model = resnet6(num_classes=num_classes)
    elif arch == "resnet50":
        model = ResNet50(num_classes=num_classes)
    else:
        assert False, "unknown model arch: " + arch

    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model))

    return model
