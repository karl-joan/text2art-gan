import os
import torch

from .models import create_model
from .options.config import BaseOptions
from .data import create_dataset
from .util.util import save_images

def cyclegan(savepath, style, dataset, identity=False, use_cpu=False, verbose=False):
    # Get options
    opt = BaseOptions().parse(verbose) 
    opt.dataroot = savepath
    opt.results_dir = savepath
    if use_cpu:
        opt.gpu_ids = []
    else:
        torch.cuda.set_device(opt.gpu_ids[0])

    model_name = ""
    
    if dataset == "birds":
        model_name += "birds2"
    else:
        model_name += "coco2"

    if style == "abstract_expressionism":
        model_name += "abs_exp"
    elif style == "impressionism":
        model_name += "imp"

    if identity == True:
        model_name += "_idt"

    opt.name = model_name

    if verbose == True:
        BaseOptions().print_options(opt)

    dataset = create_dataset(opt, verbose)  # Create a dataset given opt.dataset_mode and other options
    model = create_model(opt, verbose)      # Create a model given opt.model and other options
    model.setup(opt, verbose)               # Regular setup: load and print networks; create schedulers

    # Test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # Only apply our model to opt.num_test images.
            break

        model.set_input(data)  # Unpack data from data loader
        model.test()           # Run inference
        visuals = model.get_current_visuals()  # Get image results
        img_path = model.get_image_paths()     # Get image paths

        print(f"Applying style to image number {i+1}")
        save_images(visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
