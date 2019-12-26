import os
import torch
from .models import create_model
from .options.config import BaseOptions
from .data import create_dataset
from .util.util import save_images

def cyclegan(savepath, style, dataset, use_cpu=False):
    opt = BaseOptions().parse() # Get options
    opt.dataroot = savepath
    opt.results_dir = savepath
    if use_cpu:
        opt.gpu_ids = []
    else:
        torch.cuda.set_device(opt.gpu_ids[0])

    if style == "abstract":
        if dataset == "birds":
            opt.name = "birds2abs_exp"
        else:
            opt.name = "coco2abs_exp"
    elif style == "impressionism":
        if dataset == "birds":
            opt.name = "birds2imp"
        else:
            opt.name = "coco2imp"
    elif style == "abs2_for_coco":
        opt.name = "coco2abs_exp2"
    elif style == "gen_coco2abs_exp":
        opt.name = style
    elif style == "gen_coco2imp":
        opt.name = style
    elif style == "birds2abs_exp_idt":
        opt.name = style
    elif style == "birds2imp_idt":
        opt.name = style

    #BaseOptions().print_options(opt)

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

        print(f"Applying style to image number {i+1}")
        save_images(visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
