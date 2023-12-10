from copy import deepcopy

from yaml import (
    load, CLoader as Loader,
    dump, CDumper as Dumper
)


def load_configuration(path):
    with open(path, "r", encoding="utf-8") as fp:
        config = load(fp, Loader=Loader)
    return config


def dump_configuration(config, path):
    with open(path, "w", encoding="utf-8") as fp:
        config = dump(config, fp, Dumper=Dumper)
    return config