import matplotlib.pyplot as plt

def show_image(image):
    dpi = 80
    figsize = (image.shape[1]/float(dpi), image.shape[0]/float(dpi))
    fig = plt.figure(figsize=figsize); 
    plt.imshow(image);
    fig.show()
