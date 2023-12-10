#! /bin/env python3

import argparse
from math import log10
import os
import os.path as osp

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from torch_geometric.loader import DataLoader
from yaml import dump, CDumper as Dumper

from configuration import load_configuration
from deptree.preprocess import preprocess
from utils import dyn_load


def train_eval(train_dataset, test_dataset, config):
    # get relevant configuration options
    model_name = config["xp"]["model"]
    num_classes = int(config["dataset"]["num_classes"])  # can actually be read from Data.num_classes
    xp_dir = config["xp"]["directory"]
    device = config["xp"]["device"]
    batch_size = int(config["xp"]["hparams"]["batch_size"])
    nb_epochs = int(config["xp"]["hparams"]["nb_epochs"])
    lr = float(config["xp"]["hparams"]["lr"])

    # dynamically load model from configuration
    from_config = dyn_load(model_name, "from_config")
    # dynamically load a function to take relevant parts of data obj and send them to device
    # before the model's forward
    to_inputs = dyn_load(model_name, "to_inputs")

    model = from_config(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # TODO: dynload optimizer and loss as hparams...
    if "weighted" in config["xp"]["hparams"] and config["xp"]["hparams"]["weighted"]:
        weights = _compute_weights(train_dataset).to(device)
        criterion = torch.nn.CrossEntropyLoss(weights, reduction="mean")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # create for evaluation metrics
    _metric_collections = _init_metrics(num_classes, device)
    train_metric_collection = _metric_collections.clone(prefix="train_").to(device)
    test_metric_collection = _metric_collections.clone(prefix="test_").to(device)
    train_metrics = []
    test_metrics = []
    best_eval = None
    best_accuracy = 0.0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(0, nb_epochs):

        print(
            f"Epoch {epoch + 1}/{nb_epochs}",
            "=" * (9 + int(log10(epoch + 1)) + int(log10(nb_epochs))),
            sep="\n",
        )

        model.train()
        for data in train_loader:
            out = model(**to_inputs(data, device))
            loss = criterion(out, data.y.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred = out.argmax(dim=1)
            target = data.y.to(device)
            train_metric_collection.update(pred, target)
        print("Train: ", end="")
        metrics = train_metric_collection.compute()
        print_collection(metrics, split="train")
        train_metrics.append(metrics)
        train_metric_collection.reset()

        model.eval()
        for data in test_loader:
            out = model(**to_inputs(data, device))
            pred = out.argmax(dim=1)
            target = data.y.to(device)
            test_metric_collection.update(pred, target)
        print("Test:  ", end="")
        metrics = test_metric_collection.compute()
        print_collection(metrics, split="test")
        if metrics["test_WeightedF1Score"].item() > best_accuracy:
            best_accuracy = metrics["test_WeightedF1Score"].item()
            best_eval = metrics
            torch.save(model, f"{xp_dir}/best.pt")
        test_metrics.append(metrics)
        test_metric_collection.reset()
        print()

    print("Best results\n============")
    for name, metric in best_eval.items():
        name = name[name.index("test_") + 5:]
        print(f"{name}={metric.item():.4f}", end=" ")

    # backup configuration
    with open(osp.join(xp_dir, "config.yml"), "w", encoding="utf-8") as fp:
        dump(config, fp, Dumper=Dumper)


def _init_metrics(num_classes, device):
    return MetricCollection({
        "MacroAccuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
        "MacroPrecision": MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
        "MacroRecall": MulticlassRecall(num_classes=num_classes, average="macro").to(device),
        "MacroF1Score": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "MicroAccuracy": MulticlassAccuracy(num_classes=num_classes, average="micro").to(device),
        "MicroPrecision": MulticlassPrecision(num_classes=num_classes, average="micro").to(device),
        "MicroRecall": MulticlassRecall(num_classes=num_classes, average="micro").to(device),
        "MicroF1Score": MulticlassF1Score(num_classes=num_classes, average="micro").to(device),
        "WeightedAccuracy": MulticlassAccuracy(num_classes=num_classes, average="weighted").to(device),
        "WeightedPrecision": MulticlassPrecision(num_classes=num_classes, average="weighted").to(device),
        "WeightedRecall": MulticlassRecall(num_classes=num_classes, average="weighted").to(device),
        "WeightedF1Score": MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    })


def eval(test_dataset, config):
    model_name = config["xp"]["model"]
    num_classes = int(config["dataset"]["num_classes"])  # can actually be read from Data.num_classes
    batch_size = int(config["xp"]["hparams"]["batch_size"])
    device = config["xp"]["device"]

    model_path = osp.join(config["xp"]["directory"], "best.pt")
    if not osp.exists(model_path):
        print(f"No model found: {model_path}")
        return

    print(f"Configuration: {config}")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    to_inputs = dyn_load(model_name, "to_inputs")
    model = torch.load(model_path).to(device)
    model.eval()

    test_metric_collection = _init_metrics(num_classes, device)

    for data in test_loader:
        out = model(**to_inputs(data, device))
        pred = out.argmax(dim=1)
        target = data.y.to(device)
        test_metric_collection.update(pred, target)

    print("Test:  ", end="")
    metrics = test_metric_collection.compute()
    print_collection(metrics, split="test")
    test_metric_collection.reset()
    print()


def _compute_weights(train_dataset):
    y = np.array(train_dataset.y.tolist())
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return torch.tensor(class_weights, dtype=torch.float32)


def print_collection(collection, split="train", compute=False):
    if compute:
        items = collection.compute().items()
    else:
        items = collection.items()
    for name, metric in items:
        if split in name:
            name = name[name.index(f"{split}_") + len(split) + 1:]
        print(f"{name}={metric.item():.4f}", end=" ")
    print()


def init_result_dir(directory):
    if not osp.exists(directory):
        os.mkdir(directory)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="Task to perform. Can be either \"train\", \"eval\" or \"preprocess\"",
                        choices=["train", "eval", "preprocess"])
    parser.add_argument("target", help="Target of the task. Can be either a .yml configuration file or a directory")
    parser.add_argument("-d", "--device", help="Override configuration device with specified value")
    return parser.parse_args()


def main():
    args = _parse_args()

    config = load_configuration(args.target)
    if args.device:
        config["preprocessing"]["device"] = args.device
        config["xp"]["device"] = args.device
    print(f"Configuration: {config}")

    train_dataset, test_dataset = preprocess(
        config["dataset"]["name"],
        config["dataset"]["directory"],
        config["preprocessing"]["model"],
        config["preprocessing"]["device"],
        config["dataset"]["directed"] if "directed" in config["dataset"] else True
    )

    xp_dir = config["xp"]["directory"]

    if args.command == "train":
        if not osp.exists(xp_dir):
            os.mkdir(xp_dir)
        train_eval(train_dataset, test_dataset, config)
    elif args.command == "eval":
        eval(test_dataset, config)


if __name__ == "__main__":
    main()
