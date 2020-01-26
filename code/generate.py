import argparse
import os
from datetime import datetime

from attngan import attngan
from cyclegan import cyclegan

def parse_args():
    parser = argparse.ArgumentParser(description="Generate art from text")
    parser.add_argument("caption", help="text to generate from", type=str, metavar="\"caption\"")
    parser.add_argument("style", help="the style of the artwork", choices=["abstract_expressionism", "impressionism"], type=str)
    parser.add_argument("-d", "--dataset", dest="dataset", help="dataset to generate from (default birds)", choices=["birds", "coco"], default="birds", type=str)
    parser.add_argument("-n", "--number", dest="number", help="the number of artworks to generate (default 2)", default=2, type=int)
    parser.add_argument("-i", "--identity", dest="identity", help="set lambda_idt = 5 instead of lambda_idt = 0.5", action="store_true")
    parser.add_argument("-c", "--cpu", dest="cpu", help="use cpu", action="store_true")
    parser.add_argument("-v", "--verbose", dest="verb", help="print more details", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()

savepath = os.path.join("../results/", datetime.today().strftime("%Y-%m-%d_%H-%M-%S/"))

print("--------------- Generating images ---------------")
attngan(args.caption, args.dataset, args.number, savepath, args.cpu, args.verb)
print("---------------------- End ----------------------\n")

print("----------------- Applying style -----------------")
cyclegan(savepath, args.style, args.dataset, args.identity, args.cpu, args.verb)
print("---------------------- End ----------------------")

with open(savepath + "text", "w") as f:
    f.write(args.caption + "\n")
    f.write(args.style)
