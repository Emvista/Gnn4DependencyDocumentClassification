from copy import deepcopy
import os
import os.path as osp

import flair
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


class EmbeddedDeptreeInMemoryDataset(InMemoryDataset):
    r"""A text dataset from document classification where every document has been
        replaced by its dependency tree and nodes embeddings are initialized from flair.
        Data is stored in RAM (cpu).

    Args:
        root (str): Root directory where the dataset should be saved.
        model (str, optional)): The name of the model to be used for word embeddings.
            (default: :obj:`"distilbert-base-uncased"`)
        split (str, optional): Name of the split to load. Can be `"train"`
            or `"test"` (default: :obj:`"train"`)
        device (str, optional): Device to be used by flair to run the embedder model.
            (default: :obj:`"cpu"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(
            self,
            root,
            model="distilbert-base-uncased",
            split="train",
            device="cpu",
            transform=None,
            pre_transform=None,
            pre_filter=None,
    ):
        if split not in {"train", "test"}:
            raise ValueError(
                f"Invalid 'split' argument. Got `{split}` expected `train` or `test`"
            )
        self.split = split
        self._max_doc_id = dict()
        self.device = device
        if model:
            flair.device = device
            self.embedder = TransformerWordEmbeddings(model)
        else:
            print("No model specified for Flair embedder.")
            self.embedder = None
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        r"""The path where the dataset gets downloaded to."""
        return osp.join(self.root, self.split, "raw")

    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return ["nodes.csv", "edges.csv", "labels.csv"]

    @property
    def processed_dir(self) -> str:
        r"""The path where the processed dataset is being saved."""
        return osp.join(self.root, self.split, "processed")

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return ["data.pt"]

    # def len(self):
    #     r"""Returns the number of graphs stored in the dataset."""
    #     return self._max_doc_id[self.split]

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""

        nodes_path = osp.join(self.raw_dir, "nodes.csv")
        edges_path = osp.join(self.raw_dir, "edges.csv")
        labels_path = osp.join(self.raw_dir, "labels.csv")
        print(f"  Reading raw data for processing from:\n    {nodes_path}\n    {edges_path}")
        nodes = pd.read_csv(
            nodes_path,
            header=None,
            names=["doc_id", "node_id", "text"],
            keep_default_na=False,
        )
        edges = pd.read_csv(
            edges_path,
            header=None,
            names=["head", "tail", "relation", "doc_id"],
            keep_default_na=False,
        )
        labels = pd.read_csv(labels_path, header=None, names=["doc_id", "label"], keep_default_na=False)
        self._relation2id = {
            relation: i
            for i, relation in enumerate(sorted(edges["relation"].unique()))
        }
        self._label2id = {
            label: i
            for i, label in enumerate(sorted(labels["label"].unique()))
        }

        if not osp.exists(osp.join(self.processed_dir)):
            os.mkdir(osp.join(self.processed_dir))

        doc_ids = nodes["doc_id"].unique()
        doc_ids.sort()
        data_list = []

        for doc_id in doc_ids:
            if doc_id % 100 == 0 or doc_id == doc_ids[-1]:
                print(f"  {doc_id=}")

            doc_nodes = nodes[nodes["doc_id"] == doc_id]
            doc_words = doc_nodes["text"].tolist()
            node2id = {
                node: i
                for i, node in enumerate(doc_nodes.index.unique())  # TODO: no doc_nodes["node_id"]?
            }
            x = self._encode_text(doc_words)

            doc_edges = edges[edges["doc_id"] == doc_id]
            edge2id = {
                edge_index: i
                for i, edge_index in enumerate(doc_edges.index.unique())
            }
            edge_index = torch.empty((2, len(doc_edges)), dtype=torch.long)
            edge_type = torch.empty(len(doc_edges), dtype=torch.long)

            for index, row in doc_edges.iterrows():
                edge_index[0, edge2id[index]] = node2id[row["head"]]
                edge_index[1, edge2id[index]] = node2id[row["tail"]]
                edge_type[edge2id[index]] = self._relation2id[row["relation"]]

            data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
            _original_data = deepcopy(data)

            if self.pre_filter is not None:
                data = self.pre_filter(data)
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            label = labels[labels["doc_id"] == doc_id]["label"].item()
            y = self._encode_label(label)
            data.y = y
            # assert data.y.shape[0] == 1
            # assert y.shape[0] == 1

            data_list.append(data.to("cpu"))
            # print(torch.cuda.memory_summary("cuda:1"))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _encode_label(self, label):
        return torch.tensor([self._label2id[label]])

    def _encode_text(self, words):
        sentence = Sentence(words)
        self.embedder.embed(sentence)
        embeddings = torch.cat(
            [
                sentence[idx].embedding.unsqueeze(0)
                for idx in range(len(sentence.tokens))
            ],
            axis=0,
        )
        # sentence.clear_embeddings()
        return embeddings
