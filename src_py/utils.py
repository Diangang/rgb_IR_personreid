import matplotlib.pyplot as plt
import os, sys, torch

def show_image(image):
    dpi = 80
    figsize = (image.shape[1]/float(dpi), image.shape[0]/float(dpi))
    fig = plt.figure(figsize=figsize); 
    plt.imshow(image);
    fig.show()

def get_file_name(filepath):
    return os.path.basename(filepath).split('.')[0]
