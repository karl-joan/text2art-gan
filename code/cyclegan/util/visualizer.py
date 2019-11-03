import numpy as np
import os
import sys
from . import util

def save_images(visuals, image_path, flag, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals'.
    """
    image_dir = "./temp/"
    short_path = os.path.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    print(f"cwd: {os.getcwd()}")

    if flag == True:
        print(f"image_dir: {image_dir}")
        print(f"short_path: {short_path}")
        print(f"name: {name}")
        flag = False

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
