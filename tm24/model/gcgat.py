import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import (
    GraphConv, GATv2Conv,
    BatchNorm
)

from utils import dyn_load


class GCGAT(torch.nn.Module):

    def __init__(self,
                 num_node_features, hidden_channels, nb_heads, num_classes,
                 activation, pooling, dropout
                 ):
        super().__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.nb_heads = nb_heads
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = dyn_load(activation)
        self.pool = dyn_load(pooling)

        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.norm1 = BatchNorm(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels, nb_heads)
        self.norm3 = BatchNorm(hidden_channels * nb_heads)
        self.lin = Linear(hidden_channels * nb_heads, num_classes)

    def forward(self, **inputs):
        x = inputs["x"]
        edge_index = inputs["edge_index"]
        batch = inputs["batch"]

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def from_config(**config):
    try:
        nb_heads = config["xp"]["hparams"]["nb_heads"]
    except KeyError as e:
        nb_heads = 1
    return GCGAT(
        int(config["preprocessing"]["emb_dim"]),
        int(config["xp"]["hparams"]["hidden_channels"]),
        int(nb_heads),
        int(config["dataset"]["num_classes"]),
        config["xp"]["hparams"]["activation"],
        config["xp"]["hparams"]["pooling"],
        float(config["xp"]["hparams"]["dropout"])
    )


def to_inputs(data, device):
    return {
        "x": data.x.to(device),
        "edge_index": data.edge_index.to(device),
        "batch": data.batch.to(device)
    }