from easydict import EasyDict as edict
import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self):
        """Define the common options that are used in both training and test."""

        opt = edict()

        # Basic parameters
        opt.dataroot = "datasets/horse2zebra/testA" # Change later
        opt.name = "horse2zebra_pretrained" # Change later
        opt.gpu_ids = "0" #Remove quotes
        opt.checkpoints_dir = "./checkpoints"

        # Model parameters
        opt.model = "test" #Change later
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.ndf = 64
        opt.netD = "basic"
        opt.netG = "resnet_9blocks"
        opt.n_layers_D = 3
        opt.norm = "instance"
        opt.init_type = "normal"
        opt.init_gain = 0.02
        opt.no_dropout = True #CHange maybe

        # Dataset parameters
        opt.dataset_mode = "single" #CC
        opt.direction = "AtoB"
        opt.serial_batches = False
        opt.num_threads = 4
        opt.batch_size = 1
        opt.load_size = 256
        opt.crop_size = 256
        opt.max_dataset_size = float("inf")
        opt.preprocess = "resize_and_crop"
        opt.no_flip = True ###
        opt.display_winsize = 256

        # Additional parameters
        opt.epoch = "latest"
        opt.load_iter = 0
        opt.verbose = False
        opt.suffix = ""

        # Test parameters
        opt.ntest = float("inf")
        opt.results_dir = "./results/"
        opt.aspect_ratio = 1.0
        opt.phase = "test"
        opt.eval = False
        opt.num_test = 50 #Change later
        opt.isTrain = False

        self.isTrain = False

        return opt

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            opt = self.initialize()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        opt = model_option_setter(opt, self.isTrain)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        opt = dataset_option_setter(opt, self.isTrain)

        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        #opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

