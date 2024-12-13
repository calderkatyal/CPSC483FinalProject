"""
************************************************************
A Modifification of Heterogenous Graph Attention Network (https://arxiv.org/abs/1903.07293)
with custom attention-based metapath instance encoders

Uses code from PyTorch Geometric's Implementation of HANConv 
(https://pytorch-geometric.readthedocs.io/en/2.5.0/_modules/torch_geometric/nn/conv/han_conv.html#HANConv)
************************************************************
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
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

class HANMEConv(MessagePassing):
    """
Args:
    in_channels (int or Dict[str, int]): Size of each input sample for every
        node type, or :obj:`-1` to derive the size from the first input(s)
        to the forward method.
    out_channels (int): Size of each output sample.
    metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
        of the heterogeneous graph, *i.e.* its node and edge types given
        by a list of strings and a list of string triplets, respectively.
        See :meth:`torch_geometric.data.HeteroData.metadata` for more
        information.
    heads (int, optional): Number of multi-head attentions. (default: :obj:`1`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients, which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0.0`)
    metapath_data (Dict, optional): A dictionary containing data associated
        with metapaths, used for encoding node relationships in a
        heterogeneous graph. (default: :obj:`None`)
    node_features (Dict, optional): A dictionary mapping node types to their
        corresponding features. (default: :obj:`None`)
    metapath_encoder (str, optional): The type of encoder used for metapath
        instance encoding. Options include 'multihop', 'direct', or 'mean'.
        (default: :obj:`'multihop'`)
    **kwargs (optional): Additional arguments passed to the
        :class:`torch_geometric.nn.conv.MessagePassing` base class.
"""

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        metapath_data: Dict = None,  
        node_features: Dict = None,
        metapath_encoder: str = 'multihop',
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
        self.metapath_encoder = metapath_encoder
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
        self.W_h = nn.Parameter(torch.empty(heads, dim, dim))  
        self.W_t = nn.Parameter(torch.empty(heads, dim, dim)) 
        self.v_a = nn.Parameter(torch.empty(heads, 2*dim))   
        self.attn_scale = nn.Parameter(torch.tensor(1.0))
    
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)
        glorot(self.W_h)
        glorot(self.W_t)
        glorot(self.v_a)


    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        return_semantic_attention_weights: bool = False,
    ) -> Union[Dict[NodeType, OptTensor], Tuple[Dict[NodeType, OptTensor],
                                                Dict[NodeType, OptTensor]]]:
        device = next(self.parameters()).device

        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Project node features once at start
        for node_type in self.node_features.keys():
            x_node_dict[node_type] = self.proj[node_type](self.node_features[node_type].to(device)).view(-1, H, D)
            out_dict[node_type] = []

        # Process each metapath
        for metapath_name, data in self.metapath_data.items():
            paths = data['paths'].to(device)
            node_types = data['node_types_list']
            dst_type = node_types[0] 
            
            if self.metapath_encoder == 'mean':
                x_src = self.mean_encoder(paths, node_types, x_node_dict)
            elif self.metapath_encoder == 'multihop':
                x_src = self.multihop_encoder(paths, node_types, x_node_dict)
            elif self.metapath_encoder == 'direct':
                x_src = self.direct_attention_encoder(paths, node_types, x_node_dict)
            else:
                raise ValueError('Invalid metapath encoder')
            
            x_dst = x_node_dict[dst_type] 


            # Create edge index for instance aggregation
            edge_index = torch.stack([
                torch.arange(len(paths), device=device),
                paths[:, 0] 
            ])

            # Instance-level attention aggregation
            edge_type = f'metapath_{metapath_name}'
            lin_src = self.lin_src[edge_type].to(device)
            lin_dst = self.lin_dst[edge_type].to(device)
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
            if len(outs) > 0:  
                out, attn = group(outs, self.q.to(device), self.k_lin.to(device))
                out_dict[node_type] = out
                semantic_attn_dict[node_type] = attn

        if return_semantic_attention_weights:
            return out_dict, semantic_attn_dict

        return out_dict
    
    def direct_attention_encoder(self,
        paths: Tensor, 
        node_types: List[str],
        x_node_dict: Dict[str, Tensor]
    ) -> Tensor:
        path_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = paths[:, pos]
            pos_features = x_node_dict[node_type].to(paths.device).index_select(0, node_indices)
            path_features.append(pos_features)
        
        path_features = torch.stack(path_features, dim=1)
        
        source = torch.einsum('nphd,hde->nphe', path_features[:, 0:1], self.W_t)  # Source is target
        nodes = torch.einsum('nphd,hde->nphe', path_features, self.W_h)           # All nodes are source of attention
        
        # Compute attention scores with sigmoid instead of softmax
        scores = torch.einsum('nphe,nqhe->nqh', source, nodes)
        scores = scores / math.sqrt(self.out_channels // self.heads)
        
        attn = torch.sigmoid(scores).unsqueeze(-1)
        out = (attn * path_features).sum(dim=1)
        
        return out
        
    def multihop_encoder(self,
            paths: Tensor, 
            node_types: List[str],
            x_node_dict: Dict[str, Tensor]
        ) -> Tensor:
            # Gather and stack features
            path_features = []
            for pos, node_type in enumerate(node_types):
                node_indices = paths[:, pos]
                pos_features = x_node_dict[node_type].to(paths.device).index_select(0, node_indices)
                path_features.append(pos_features)

            gamma = .4
            path_features = torch.stack(path_features, dim=1)  
            path_length = path_features.size(1)
            path_features = torch.cat([path_features[:, :1], path_features], dim=1) 

            # Prepare pairs of source and target nodes
            outer_nodes = torch.einsum('nphd,hde->nphe', path_features[:, :-1], self.W_h)  # Target nodes (being attended to)
            inner_nodes = torch.einsum('nphd,hde->nphe', path_features[:, 1:], self.W_t)   # Source nodes (doing attending)

            # Compute attention scores
            concat = torch.cat([inner_nodes, outer_nodes], dim=-1)
            logits = torch.einsum('nphe,he->nph', torch.tanh(concat), self.v_a)

            del inner_nodes, outer_nodes, concat
            torch.cuda.empty_cache()

            attn_logits = F.leaky_relu(logits, self.negative_slope)
            del logits
            torch.cuda.empty_cache()

            attn_weights = torch.sigmoid(attn_logits)  
            
            """    
            if self.training:
                idx = torch.randint(0, paths.size(0), (1,)).item()
                print(f"\nInstance {idx} Details:")
                print(f"Path: {paths[idx].tolist()}")
                print(f"Attention Logits: {attn_logits[idx].cpu().detach().numpy()}")
                print(f"Final Weights: {attn_weights[idx].cpu().detach().numpy()}")
            """ 

            del attn_logits
            torch.cuda.empty_cache()

            # Split and handle self-attention
            self_attn = attn_weights[:, 0:1]  
            path_attn = attn_weights[:, 1:]  
            chain_attn = torch.cat([self_attn, torch.cumprod(path_attn, dim=1)], dim=1)

            hop_weights = gamma * torch.pow(1 - gamma, torch.arange(path_length, device=paths.device))
            chain_attn = chain_attn * hop_weights.view(1, -1, 1)

            out = (chain_attn.unsqueeze(-1) * path_features[:, 1:]).sum(dim=1)

            del chain_attn, hop_weights
            torch.cuda.empty_cache()

            return out


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
        # Gather features for all nodes in paths
        path_features = []
        for pos, node_type in enumerate(node_types):
            node_indices = paths[:, pos]
            pos_features = x_node_dict[node_type].to(paths.device).index_select(0, node_indices.to(paths.device))
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