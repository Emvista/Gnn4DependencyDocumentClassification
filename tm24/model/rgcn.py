import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import (
    RGCNConv,
    RGATConv,
    BatchNorm
)

from utils import dyn_load


class RGCN(torch.nn.Module):

    def __init__(self,
                 num_node_features, hidden_channels, num_relations, num_classes,
                 activation, pooling, dropout
                 ):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, hidden_channels, num_relations)
        self.norm1 = BatchNorm(hidden_channels)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.norm2 = BatchNorm(hidden_channels)
        self.conv3 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.norm3 = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.activation = dyn_load(activation)
        self.pool = dyn_load(pooling)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, batch):
        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = self.norm2(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_type)
        x = self.norm3(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


def from_config(**config):
    return RGCN(
        int(config["preprocessing"]["emb_dim"]),
        int(config["xp"]["hparams"]["hidden_channels"]),
        int(config["dataset"]["num_relations"]),
        int(config["dataset"]["num_classes"]),
        config["xp"]["hparams"]["activation"],
        config["xp"]["hparams"]["pooling"],
        float(config["xp"]["hparams"]["dropout"])
    )


def to_inputs(data, device):
    return {
        "x": data.x.to(device),
        "edge_index": data.edge_index.to(device),
        "edge_type": data.edge_type.to(device),
        "batch": data.batch.to(device)
    }