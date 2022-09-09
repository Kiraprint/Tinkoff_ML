import pickle as pkl
import argparse
from pathlib import Path
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prefix', type=str, required=False)
    parser.add_argument('--length', type=int, required=True)
    args = parser.parse_args()
    with open(args.model,'rb') as f:
        model = pkl.load(f)
        model.random_seed = 42
    output = model.generate(args.length, args.prefix)
    print(output)
