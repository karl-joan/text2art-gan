import os
from .models import create_model
from .options.config import BaseOptions
from .data import create_dataset
from .util.util import save_images

def cyclegan(savepath, cpu=False):
    opt = BaseOptions().parse() # Get options
    opt.dataroot = savepath
    opt.results_dir = savepath

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print(img_path)

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
