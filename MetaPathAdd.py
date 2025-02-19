"""
*******************************************************************
Modifification of torch_geometric.transforms.AddMetaPaths which stores
node types and metapath instances in a dictionary
*******************************************************************
"""
import warnings
from typing import List, Optional, Tuple, Dict, Union, cast

import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import coalesce, degree
from tqdm import tqdm

class MetaPathAdd(BaseTransform):
    r"""Adds additional edge types to a
    :class:`~torch_geometric.data.HeteroData` object between the source node
    type and the destination node type of a given :obj:`metapath`, as described
    in the `"Heterogenous Graph Attention Networks"
    <https://arxiv.org/abs/1903.07293>`_ paper
    (functional name: :obj:`add_metapaths`).

    Meta-path based neighbors can exploit different aspects of structure
    information in heterogeneous graphs.
    Formally, a metapath is a path of the form

    .. math::

        \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
        \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

    in which :math:`\mathcal{V}_i` represents node types, and :math:`R_j`
    represents the edge type connecting two node types.
    The added edge type is given by the sequential multiplication  of
    adjacency matrices along the metapath, and is added to the
    :class:`~torch_geometric.data.HeteroData` object as edge type
    :obj:`(src_node_type, "metapath_*", dst_node_type)`, where
    :obj:`src_node_type` and :obj:`dst_node_type` denote :math:`\mathcal{V}_1`
    and :math:`\mathcal{V}_{\ell}`, repectively.

    In addition, a :obj:`metapath_dict` object is added to the
    :class:`~torch_geometric.data.HeteroData` object which maps the
    metapath-based edge type to its original metapath.

    .. code-block:: python

        from torch_geometric.datasets import DBLP
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import AddMetaPaths

        data = DBLP(root)[0]
        # 4 node types: "paper", "author", "conference", and "term"
        # 6 edge types: ("paper","author"), ("author", "paper"),
        #               ("paper, "term"), ("paper", "conference"),
        #               ("term, "paper"), ("conference", "paper")

        # Add two metapaths:
        # 1. From "paper" to "paper" through "conference"
        # 2. From "author" to "conference" through "paper"
        metapaths = [[("paper", "conference"), ("conference", "paper")],
                     [("author", "paper"), ("paper", "conference")]]
        data = AddMetaPaths(metapaths)(data)

        print(data.edge_types)
        [("author", "to", "paper"), ("paper", "to", "author"),
             ("paper", "to", "term"), ("paper", "to", "conference"),
             ("term", "to", "paper"), ("conference", "to", "paper"),
             ("paper", "metapath_0", "paper"),
             ("author", "metapath_1", "conference")]

        print(data.metapath_dict)
        {("paper", "metapath_0", "paper"): [("paper", "conference"),
                                                ("conference", "paper")],
             ("author", "metapath_1", "conference"): [("author", "paper"),
                                                      ("paper", "conference")]}

    Args:
        metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
            by a list of lists of
            :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
        drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
            edge types will be dropped. (default: :obj:`False`)
        keep_same_node_type (bool, optional): If set to :obj:`True`, existing
            edge types between the same node type are not dropped even in case
            :obj:`drop_orig_edge_types` is set to :obj:`True`.
            (default: :obj:`False`)
        drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
            will drop node types not connected by any edge type.
            (default: :obj:`False`)
        max_sample (int, optional): If set, will sample at maximum
            :obj:`max_sample` neighbors within metapaths. Useful in order to
            tackle very dense metapath edges. (default: :obj:`None`)
        weighted (bool, optional): If set to :obj:`True`, computes weights for
            each metapath edge and stores them in :obj:`edge_weight`. The
            weight of each metapath edge is computed as the number of metapaths
            from the start to the end of the metapath edge.
            (default :obj:`False`)
    """
    def __init__(
        self,
        metapaths: List[List[EdgeType]],
        drop_orig_edge_types: bool = False,
        keep_same_node_type: bool = False,
        drop_unconnected_node_types: bool = False,
        max_sample: Optional[int] = None,
        weighted: bool = False,
        **kwargs: bool,
    ) -> None:
        if 'drop_orig_edges' in kwargs:
            warnings.warn("'drop_orig_edges' is dprecated. Use "
                          "'drop_orig_edge_types' instead")
            drop_orig_edge_types = kwargs['drop_orig_edges']

        if 'drop_unconnected_nodes' in kwargs:
            warnings.warn("'drop_unconnected_nodes' is dprecated. Use "
                          "'drop_unconnected_node_types' instead")
            drop_unconnected_node_types = kwargs['drop_unconnected_nodes']

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_node_types = drop_unconnected_node_types
        self.max_sample = max_sample
        self.weighted = weighted

    def forward(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        edge_types = data.edge_types  # Save original edge types.
        data.metapath_dict = {}
        metapath_data = {}  # Initialize the dictionary to store paths and features.

        for j, metapath in enumerate(tqdm(self.metapaths, desc='Processing metapaths')):
            for edge_type in metapath:
                assert data._to_canonical(edge_type) in edge_types

            edge_type = metapath[0]
            edge_index, edge_weight = self._edge_index(data, edge_type)

            if self.max_sample is not None:
                edge_index, edge_weight = self._sample(edge_index, edge_weight)

            # Initialize the paths with the edges of the first relation
            edge_index_tensor = edge_index.as_tensor()
            src_nodes = edge_index_tensor[0].tolist()
            dst_nodes = edge_index_tensor[1].tolist()
            current_paths = [[src, dst] for src, dst in zip(src_nodes, dst_nodes)]

            for i, edge_type in enumerate(tqdm(metapath[1:], desc=f'Processing metapath {j} edge types')):
                edge_index2, edge_weight2 = self._edge_index(data, edge_type)
                edge_index2_tensor = edge_index2.as_tensor()
                edge_index2_src = edge_index2_tensor[0].tolist()
                edge_index2_dst = edge_index2_tensor[1].tolist()
                edge_index2_dict = {}
                for src, dst in zip(edge_index2_src, edge_index2_dst):
                    edge_index2_dict.setdefault(src, []).append(dst)

                new_paths = []
                for path in tqdm(current_paths, desc='Processing current paths'):
                    last_node = path[-1]
                    if last_node in edge_index2_dict:
                        for next_node in edge_index2_dict[last_node]:
                            new_paths.append(path + [next_node])
                current_paths = new_paths

                # Update edge_index based on the new paths
                src_nodes = [path[0] for path in current_paths]
                dst_nodes = [path[-1] for path in current_paths]
                edge_index = EdgeIndex(
                    torch.tensor([src_nodes, dst_nodes], dtype=torch.long),
                    sparse_size=(data[metapath[0][0]].num_nodes,
                                data[metapath[-1][-1]].num_nodes),
                )

                if not self.weighted:
                    edge_weight = None

                if self.max_sample is not None:
                    edge_index, edge_weight = self._sample(edge_index, edge_weight)

            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = edge_index
            if self.weighted:
                data[new_edge_type].edge_weight = edge_weight
            data.metapath_dict[new_edge_type] = metapath
           # Map metapath index to name 
            metapath_key = f'metapath_{j}'
            metapath_name = metapath_key 


            metapath_data[metapath_name] = {
                'node_types_list': [metapath[0][0]] + [e[2] for e in metapath],
                'paths':torch.tensor(current_paths, dtype=torch.long),
            }
            

        postprocess(data, edge_types, self.drop_orig_edge_types,
                    self.keep_same_node_type, self.drop_unconnected_node_types)

        return data, metapath_data  # Return the data and the metapath_data dictionary

    def _edge_index(
        self,
        data: HeteroData,
        edge_type: EdgeType,
    ) -> Tuple[EdgeIndex, Optional[Tensor]]:

        edge_index = EdgeIndex(
            data[edge_type].edge_index,
            sparse_size=data[edge_type].size(),
        )
        edge_index, perm = edge_index.sort_by('row')

        if not self.weighted:
            return edge_index, None

        edge_weight = data[edge_type].get('edge_weight')
        if edge_weight is not None:
            assert edge_weight.dim() == 1
            edge_weight = edge_weight[perm]

        return edge_index, edge_weight

    def _sample(
        self,
        edge_index: EdgeIndex,
        edge_weight: Optional[Tensor],
    ) -> Tuple[EdgeIndex, Optional[Tensor]]:

        deg = degree(edge_index[0], num_nodes=edge_index.get_sparse_size(0))
        prob = (self.max_sample * (1. / deg))[edge_index[0]]
        mask = torch.rand_like(prob) < prob

        edge_index = cast(EdgeIndex, edge_index[:, mask])
        assert isinstance(edge_index, EdgeIndex)
        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return edge_index, edge_weight

def postprocess(
    data: HeteroData,
    edge_types: List[EdgeType],
    drop_orig_edge_types: bool,
    keep_same_node_type: bool,
    drop_unconnected_node_types: bool,
) -> None:

    if drop_orig_edge_types:
        for i in edge_types:
            if keep_same_node_type and i[0] == i[-1]:
                continue
            else:
                del data[i]

    # Remove nodes not connected by any edge type:
    if drop_unconnected_node_types:
        new_edge_types = data.edge_types
        node_types = data.node_types
        connected_nodes = set()
        for i in new_edge_types:
            connected_nodes.add(i[0])
            connected_nodes.add(i[-1])
        for node in node_types:
            if node not in connected_nodes:
                del data[node]