from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import PairTensor  # noqa
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType, OptTensor
from torch_geometric.utils import softmax


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
    """The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/abs/1903.07293>`_ paper.

    .. note::

        For an example of using HANConv, see `examples/hetero/han_imdb.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/han_imdb.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        metapath_data: Dict = None,  # Added
        node_features: Dict = None,   # Added
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
        self.metapath_data = metapath_data
        self.node_features = node_features

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for metapath_name in metapath_data.keys():
            edge_type = f'metapath_{metapath_name}'
            self.lin_src[edge_type] = nn.Parameter(torch.empty(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.empty(1, heads, dim))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        return_semantic_attention_weights: bool = False,
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor],
                                                Dict[NodeType, OptTensor]]]:
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Project node features once at start
        for node_type in self.node_features.keys():
            x_node_dict[node_type] = self.proj[node_type](self.node_features[node_type]).view(-1, H, D)
            out_dict[node_type] = []

        # Process each metapath
        for metapath_name, data in self.metapath_data.items():
            paths = data['paths']
            node_types = data['node_types_list']
            dst_type = node_types[0] 
            
            # Encode metapath instances
            x_src = self.mean_encoder(paths, node_types, x_node_dict)  # [num_instances, heads, dim]
            x_dst = x_node_dict[dst_type] 


            # Create edge index for instance aggregation
            edge_index = torch.stack([
                torch.arange(len(paths), device=paths.device),
                paths[:, 0] 
            ])


            # Instance-level attention aggregation
            edge_type = f'metapath_{metapath_name}'
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            
            # Propagate to aggregate instances
            out = self.propagate(edge_index, x=(x_src, x_dst),
                     alpha=(alpha_src, alpha_dst))
            
            out = F.relu(out)
            out_dict[dst_type].append(out)

        # Semantic attention across metapaths
        semantic_attn_dict = {}
        for node_type, outs in out_dict.items():
            if len(outs) > 0:  # Only process if we have any metapaths for this node type
                out, attn = group(outs, self.q, self.k_lin)
                out_dict[node_type] = out
                semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return out_dict, semantic_attn_dict

        return out_dict
    def mean_encoder(self,
        paths: Tensor,
        node_types: List[str],
        x_node_dict: Dict[str, Tensor]
    ) -> Tensor:
        """
        Encodes metapath instances using mean pooling over node features.

        Args:
            paths (Tensor): Tensor of shape [num_instances, path_length], 
                            containing indices of nodes in the metapath.
            node_types (List[str]): List of node types in the metapath.
            x_node_dict (Dict[str, Tensor]): Dictionary mapping node types to 
                                            their projected feature tensors.

        Returns:
            Tensor: Encoded metapath instance features of shape 
                    [num_instances, heads, dim].
        """
        # Gather projected features for all nodes in paths
        path_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = paths[:, pos]
            pos_features = x_node_dict[node_type].index_select(0, node_indices)
            path_features.append(pos_features)
        # Stack and mean pool [num_instances, path_length, heads, dim]
        path_features = torch.stack(path_features, dim=1)
        return torch.mean(path_features, dim=1)  # [num_instances, heads, dim]


    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')