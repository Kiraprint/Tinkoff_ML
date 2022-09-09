import argparse
from pathlib import Path
import sys
import numpy as np
import codecs as cs  # for better unicode support
import re
from typing import List
from functools import partial
from multiprocessing import Pool, cpu_count
import pickle as pkl


def correct(x: str) -> str:
    x = ' '.join(x.split())
    x = re.sub(r'[^\w\s]', '', x)
    x = re.sub(r'[0-9]', '', x).lower()
    return x


def preprocess(data: List[str]) -> List[str]:
    with Pool(cpu_count()) as p:  # multiprocessing
        data = p.map(correct, data)
    return data


def get_prefix_dict(text: str, n: int) -> dict:
    prefix_dict = PrefixDict()
    text = text.split()
    for i in range(n, len(text)):
        key = ' '.join(text[i-n:i])
        prefix_dict[key] = prefix_dict.get(
            key, PrefixDict()) + PrefixDict({text[i]: 1})
    return prefix_dict


class PrefixDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __add__(self, other):
        for key, value in other.items():
            zero = PrefixDict() if type(value) == PrefixDict else 0
            self[key] = self.get(key, zero) + value
        return self

    def __iadd__(self, other):
        return self.__add__(other)


class Model:
    def __init__(self, n: int = 2, random_seed: int = None) -> None:  # n - length of prefix
        self.prefix_dict = PrefixDict()
        self.n = n
        self.K = 0
        self.alpha = 0.01
        self.random_seed = random_seed

    def fit(self, data: List[str]) -> None:
        gpf = partial(get_prefix_dict, n=self.n)
        with Pool(cpu_count()) as p:
            self.prefix_dict = sum(p.map(gpf, data), start=PrefixDict())
        self.K = sum(sum(i.values()) for i in self.prefix_dict.values())
        # number of all possible postfixes (for lindstone smoothing)
        self.K = self.K**self.n - self.K

    def generate(self, length: int, prefix: str = None) -> str:
        if self.random_seed:
            np.random.seed(self.random_seed)
        prefix = prefix if prefix else np.random.choice(
            list(self.prefix_dict.keys()), 1)[0]
        string = prefix
        for _ in range(length):
            prefix_dict = self.prefix_dict[prefix]
            N = len(prefix_dict.keys())
            lindstone_smooth = (
                np.array(list(prefix_dict.values()))+self.alpha)/(N + self.alpha*self.K)
            normalized_lindstone = lindstone_smooth/(sum(lindstone_smooth))
            postfix = np.random.choice(
                list(prefix_dict.keys()), 1, p=(normalized_lindstone))[0]
            string += ' ' + postfix
            prefix = ' '.join(string.split()[-self.n:])
        return string

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pkl.dump(self, f)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--model', type=str, default=r'model.pkl')
    args = parser.parse_args()

    MODEL_SAVE_PATH = Path(args.model)

    # Read data from dir/terminal
    if args.input_dir:
        input_dir = Path(args.input_dir)
        data = []
        for f in input_dir.iterdir():
            if f.is_file():
                try:
                    with cs.open(f, 'r', encoding='cp1251') as text:
                        data.append(text.read())
                except UnicodeDecodeError:
                    try:
                        with cs.open(f, 'r', encoding='utf-8-sig') as text:
                            data.append(text.read())
                    except UnicodeDecodeError:
                        print(f'Error while reading {f}')    
    else:
        data = sys.stdin.readlines()

    # ML part (preprocess, train, save model)
    data = preprocess(data)
    model = Model()
    model.fit(data)
    model.save(MODEL_SAVE_PATH)
