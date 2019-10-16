import argparse
import os
#os.chdir("attngan/")
print(os.getcwd())
from attngan import attngan

def parse_args():
    parser = argparse.ArgumentParser(description="Generate art from text")
    parser.add_argument("caption", help="text to generate from", type=str)
    parser.add_argument("style", help="the style of the artwork", type=str)
    parser.add_argument("-d", "--dataset", dest="dataset", help="dataset to generate from (default birds)", choices=["birds", "coco"], default="birds", type=str)
    parser.add_argument("-c", "--cpu", dest="cpu", help="use cpu (default false)", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()
attngan(args.caption, args.dataset)
#print(args.caption, args.dataset)
