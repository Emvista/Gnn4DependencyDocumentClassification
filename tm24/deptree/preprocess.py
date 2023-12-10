#! /bin/env python3

import argparse
import csv
import logging
import os
import os.path as osp

import datasets
from flair.datasets import TREC_50, TREC_6
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import stanza
from torch_geometric.transforms import ToUndirected

from .dataset import EmbeddedDeptreeInMemoryDataset


def dependency_parsing(df_train, df_test, device=None, chunk_size=100):
    """Apply dependency parsing on raw data.

    Args:
        df_train (DataFrame): Training data
        df_test (DataFrame): Test data
        device (str, optional): Device to be used to computation, if any. Default: None"

    Returns:
        dict: A dictionary with processed dataset items as Stanza documents and associated labels
    """
    if device is None or device == "cpu":
        use_gpu = False
        device = None
    else:
        use_gpu = True
        device = device

    stanza.download("en")
    nlp = stanza.Pipeline(
        "en", processors="tokenize,pos,lemma,depparse",
        use_gpu=use_gpu, device=device
    )

    documents = dict()

    for split_name, df_split in [("train", df_train), ("test", df_test)]:

        print(f"Processing {split_name}: {len(df_split)} documents...")

        texts = df_split["text"].tolist()
        labels = df_split["label"].tolist()

        documents[split_name] = {"document": [], "label": []}
        for i in range(0, len(texts), chunk_size):
            # print(f"\tdocument chunk: [{i}:{i+chunk_size}]")
            documents[split_name]["document"].extend(nlp.bulk_process(texts[i:i + chunk_size]))
            documents[split_name]["label"].extend(labels[i:i + chunk_size])
        # documents[split_name] = {
        #     "document": nlp.bulk_process(texts),
        #     "label": df_split["label"].tolist()
        # }

    print("Stanza processing done.")
    return documents


def save_dependency_graphs(documents, directory):
    """Saves dependency graphs of a dataset in CSV files.

    Args:
        documents (dict): Parsed dataset
        directory (str): Directoy when dependency graphs are to be saved
    """
    word2nid = {"train": dict(), "test": dict()}

    for split_name, split in documents.items():
        nodes_path = f"{directory}/{split_name}/raw/nodes.csv"
        print(f"Saving syntatic nodes to {nodes_path}")
        with open(nodes_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            node_id = 0
            for doc_id, document in enumerate(split["document"]):
                for sent_id, sentence in enumerate(document.sentences):
                    for word_id, word in enumerate(sentence.words):
                        writer.writerow([doc_id, node_id, word.text])
                        word2nid[split_name][(doc_id, sent_id, word_id)] = node_id
                        node_id += 1

    for split_name, split in documents.items():
        edges_path = f"{directory}/{split_name}/raw/edges.csv"
        print(f"Saving syntatic edges to {edges_path}")
        with open(edges_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            for doc_id, document in enumerate(split["document"]):
                for sent_id, sentence in enumerate(document.sentences):
                    for word_id, word in enumerate(sentence.words):
                        if word.deprel == "root":
                            continue
                        writer.writerow([
                            word2nid[split_name][(doc_id, sent_id, word_id)],  # head/governor
                            word2nid[split_name][(doc_id, sent_id, word.head - 1)],  # tail/dependant
                            word.deprel,  # relation
                            doc_id,
                        ])

    for split_name, split in documents.items():
        labels_path = f"{directory}/{split_name}/raw/labels.csv"
        print(f"Saving document to label mapping to {labels_path}")
        with open(labels_path, "w", newline="") as fp:
            writer = csv.writer(fp)
            for doc_id, label in enumerate(split["label"]):
                writer.writerow([doc_id, label])


def embed(directory, model="distilbert-base-uncased", directed=True, device="cpu"):
    transformation = ToUndirected() if not directed else None
    train_dataset = EmbeddedDeptreeInMemoryDataset(directory, model, device=device, split="train",
                                                   pre_transform=transformation)
    test_dataset = EmbeddedDeptreeInMemoryDataset(directory, model, device=device, split="test",
                                                  pre_transform=transformation)
    return (train_dataset, test_dataset)


def init_dataset_dir(directory):
    if not osp.exists(directory):
        os.mkdir(directory)
    for split in ["train", "test"]:
        split_dir = osp.join(directory, split)
        if not osp.exists(split_dir):
            os.mkdir(split_dir)
            for state in ["raw", "processed"]:
                state_dir = osp.join(split_dir, state)
                if not osp.exists(state_dir):
                    os.mkdir(state_dir)


def _corpus_to_dataframe(corpus):
    data = []

    for sentence in corpus:
        label = sentence.labels[0].value if sentence.labels else None
        text = sentence.to_plain_string()
        data.append((text, label))

    return pd.DataFrame(data, columns=["Text", "Label"])


def _load_trec(dataset_name):
    if dataset_name == "trec_6":
        corpus = TREC_6()
    elif dataset_name == "trec_50":
        corpus = TREC_50()

    df_train = _corpus_to_dataframe(corpus.train)
    df_dev = _corpus_to_dataframe(corpus.dev)
    df_test = _corpus_to_dataframe(corpus.test)

    df_test.columns = ["text", "label"]
    df_train.columns = ["text", "label"]
    df_dev.columns = ["text", "label"]

    df_train = pd.concat([df_train, df_dev], axis=0)

    return df_train, df_test


def _load_agnews():
    dataset = datasets.load_dataset("ag_news")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    return df_train, df_test


def _bunch_to_dataframe(corpus):
    data = [
        (text, corpus.target_names[idx])
        for text, idx in zip(corpus.data, corpus.target)
    ]

    df = pd.DataFrame(data, columns=["text", "label"])
    df["length"] = df['text'].str.len()

    return df


def _load_20ng():
    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    df_train = _bunch_to_dataframe(train)
    df_train.drop(9247, axis="index", inplace=True)

    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    df_test = _bunch_to_dataframe(test)

    return df_train, df_test


def _load_arxiv(path="../dataset_arxiv.csv"):
    df_arxiv = pd.read_csv(path)
    df_arxiv.columns = ["text", "label"]
    return train_test_split(df_arxiv, train_size=0.75)


def load_dataset(dataset_name):
    if dataset_name == "ag_news":
        return _load_agnews()
    elif dataset_name in ["trec_6", "trec_50"]:
        return _load_trec(dataset_name)
    elif dataset_name == "20ng":
        return _load_20ng()
    elif dataset_name == "arxiv":
        return _load_arxiv()
    raise KeyError(f"Unknown dataset: {dataset_name}")


def preprocess(dataset, directory, model, device, directed=True):
    init_dataset_dir(directory)

    if not _parsing_done(directory):
        df_train, df_test = load_dataset(dataset)
        documents = dependency_parsing(df_train, df_test, device=device)
        del dataset, df_train, df_test
        save_dependency_graphs(documents, directory)
        del documents
    else:
        print("Parsing already done. Skipping.")

    return embed(
        directory,
        model if not _embedding_done(directory) else None,
        directed,
        device
    )


# ===========================================================================


def _parsing_done(directory):
    return all(
        osp.exists(osp.join(directory, split, "raw", f))
        for split in ["train", "test"]
        for f in ["nodes.csv", "edges.csv", "labels.csv"]
    )


def _embedding_done(directory):
    return all(
        osp.exists(osp.join(directory, split, "processed", "data.pt"))
        for split in ["train", "test"]
    )
