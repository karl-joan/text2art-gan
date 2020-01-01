import argparse
import os
from datetime import datetime

from attngan import attngan
from cyclegan import cyclegan

def parse_args():
    parser = argparse.ArgumentParser(description="Generate art from text")
    parser.add_argument("caption", help="text to generate from", type=str, metavar="\"caption\"")
    parser.add_argument("style", help="the style of the artwork", choices=["abstract", "impressionism", "abs2_for_coco", "gen_coco2abs_exp", "gen_coco2imp", "birds2abs_exp_idt", "birds2imp_idt"], type=str)
    parser.add_argument("-d", "--dataset", dest="dataset", help="dataset to generate from (default birds)", choices=["birds", "coco"], default="birds", type=str)
    parser.add_argument("-n", "--number", dest="number", help="the number of artworks to generate (default 2)", default=2, type=int)
    parser.add_argument("-c", "--cpu", dest="cpu", help="use cpu (default false)", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verb", help="print more details", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()

if args.style == "abs2_for_coco" and args.dataset == "birds":
    raise Exception("abs2_for_coco is only for coco dataset")

savepath = os.path.join("../results/", datetime.today().strftime("%Y-%m-%d_%H-%M-%S/"))

print("--------------- Generating images ---------------")
attngan(args.caption, args.dataset, args.number, savepath, args.cpu, args.verb)
print("---------------------- End ----------------------\n")

print("----------------- Applying style -----------------")
cyclegan(savepath, args.style, args.dataset, args.cpu, args.verb)
print("---------------------- End ----------------------")
