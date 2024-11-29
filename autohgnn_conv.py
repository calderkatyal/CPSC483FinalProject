# autohgnn_conv.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax
from typing import Dict, List, Optional, Tuple, Union

def group(
    xs: List[Tensor],
    q: nn.Parameter,
    k_lin: nn.Module,
) -> Tuple[OptTensor, OptTensor]:

    if len(xs) == 0:
        return None, None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        if out.numel() == 0:
            return out.view(0, out.size(-1)), None
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out, attn

class AutoHGNNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.empty(1, out_channels))

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.empty(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.empty(1, heads, dim))

        # Store node type sequences for each metapath
        self.metapath_types = {
            'metapath_0': ['movie', 'director', 'movie'],  # MDM
            'metapath_1': ['movie', 'actor', 'movie'],     # MAM
            'metapath_2': ['movie', 'keyword', 'movie']    # MKM
        }

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)

    def mean_encode_metapath(self, x_dict, metapath_indices, mp_type) -> Tensor:
        """Mean encode the nodes in a metapath instance, preserving sequence info."""
        # Get node types for this metapath
        node_types = self.metapath_types[mp_type]
        
        # Get features for each position in sequence
        seq_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = metapath_indices[:, pos]
            pos_features = x_dict[node_type][node_indices]  # (num_instances, channels)
            seq_features.append(pos_features)
            
        # Stack to get (num_instances, path_length, channels)
        instance_features = torch.stack(seq_features, dim=1)
        # Mean over path positions
        instance_repr = torch.mean(instance_features, dim=1)
        return instance_repr

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        metapath_dict: Dict[str, Tensor],
        return_semantic_attention_weights: bool = False,
    ):
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Project features
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Process each metapath
        for edge_type, edge_index in edge_index_dict.items():
            src_type, mp_type, dst_type = edge_type
            edge_type_str = '__'.join(edge_type)
            
            # Get metapath instance embeddings using mean encoding
            metapath_indices = metapath_dict[mp_type]
            x_src = self.mean_encode_metapath(x_node_dict, metapath_indices, mp_type)
            x_src = x_src.view(-1, H, D)
            x_dst = x_node_dict[dst_type]

            # Rest follows HAN's logic
            lin_src = self.lin_src[edge_type_str]
            lin_dst = self.lin_dst[edge_type_str]
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            
            out = self.propagate(edge_index, x=(x_src, x_dst),
                               alpha=(alpha_src, alpha_dst))
            out = F.relu(out)
            out_dict[dst_type].append(out)

        # Semantic attention
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            out, attn = group(outs, self.q, self.k_lin)
            out_dict[node_type] = out
            semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return out_dict, semantic_attn_dict

        return out_dict

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)