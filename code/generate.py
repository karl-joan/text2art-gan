import argparse
from attngan import attngan, cfg
from cyclegan import cyclegan

def parse_args():
    parser = argparse.ArgumentParser(description="Generate art from text")
    parser.add_argument("caption", help="text to generate from", type=str, metavar="\"caption\"")
    parser.add_argument("style", help="the style of the artwork", type=str)
    parser.add_argument("-d", "--dataset", dest="dataset", help="dataset to generate from (default birds)", choices=["birds", "coco"], default="birds", type=str)
    parser.add_argument("-n", "--number", dest="number", help="the number of artworks to generate (default 2)", default=2, type=int)
    parser.add_argument("-c", "--cpu", dest="cpu", help="use cpu (default false)", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()

if args.cpu == True:
    cfg.CUDA = False

savepath = attngan(args.caption, args.dataset, args.number)
#savepath = "../results/2019-10-17_19-46-20"

cyclegan(savepath)
