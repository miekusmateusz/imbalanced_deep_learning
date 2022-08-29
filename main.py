#!/usr/bin/python3
import sys

from train_experiments.train_minibatch_quadruplets import train_minibatch_quadruplets
from train_experiments.train_minibatch_triplets import train_minibatch_triplets
from utils import parse_configuration


def main(argv):
    print(argv)

    if len(argv) != 1:
        raise Exception("Input should only have one argument - path to the configuration file!")

    path = argv[0]

    configuration = parse_configuration(path)

    train_type = configuration['type']

    if train_type == "triplet":
        train_minibatch_triplets(path)
    elif train_type == "quadruplet":
        train_minibatch_quadruplets(path)
    else:
        raise Exception("wrong type specified in configuration")


if __name__ == "__main__":
    main(sys.argv[1:])
