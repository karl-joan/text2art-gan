import argparse
from attngan import attngan
from cyclegan import cyclegan

def parse_args():
    parser = argparse.ArgumentParser(description="Generate art from text")
    parser.add_argument("caption", help="text to generate from", type=str, metavar="\"caption\"")
    parser.add_argument("style", help="the style of the artwork", choices=["abstract", "impressionism", "abs2_for_coco"], type=str)
    parser.add_argument("-d", "--dataset", dest="dataset", help="dataset to generate from (default birds)", choices=["birds", "coco"], default="birds", type=str)
    parser.add_argument("-n", "--number", dest="number", help="the number of artworks to generate (default 2)", default=2, type=int)
    parser.add_argument("-c", "--cpu", dest="cpu", help="use cpu (default false)", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()

if args.style == "abs2_for_coco" and args.dataset == "birds":
    raise Exception("abs2_for_coco is only for coco dataset")

if args.style == "impressionism" and args.dataset == "coco":
    raise Exception("impressionism style is currently only available for birds dataset")

savepath = attngan(args.caption, args.dataset, args.number, args.cpu)

cyclegan(savepath, args.style, args.dataset, args.cpu)
