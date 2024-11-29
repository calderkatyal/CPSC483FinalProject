import torch
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Union
import torch_geometric as pyg
from torch_scatter import scatter_mean
from torch import nn
import torch.nn.functional as F
from load_imdb import load_imdb
from autohgnn_conv import AutoHGNNConv
import os.path as osp

def extract_all_metapath_instances(hg, metapath):
    """Extract ALL possible node sequences following a specific metapath.
    Returns sequences as lists of node indices."""
    instances = []
    src_type = metapath[0][0]
    
    # For each start node
    for start_node in range(hg[src_type].num_nodes):
        current_nodes = {start_node}
        paths = [[start_node]]
        
        # Follow the metapath
        for src_type, edge_type, dst_type in metapath:
            edge_index = hg[src_type, edge_type, dst_type].edge_index
            new_paths = []
            
            # For each current path
            for path in paths:
                last_node = path[-1]
                # Find all neighbors
                neighbors = edge_index[1][edge_index[0] == last_node]
                
                # Add all possible continuations
                for neighbor in neighbors:
                    new_paths.append(path + [neighbor.item()])
            
            paths = new_paths
        
        # Add all complete paths to instances
        instances.extend(paths)
    
    return instances

def convert_instances_to_tensors(instances):
    """Convert metapath instances to MAGNN's tensor format."""
    if not instances:
        return torch.empty((0, 0), dtype=torch.long)
    instance_tensor = torch.tensor(instances, dtype=torch.long)
    return instance_tensor

def process_metapath_edges(instances: List[List[int]], num_nodes: int) -> torch.Tensor:
    """Convert instances to unique edge indices."""
    if not instances:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Convert to set of tuples to remove duplicates
    unique_edges = set((seq[0], seq[-1]) for seq in instances)
    
    # Convert back to tensor
    edge_index = torch.tensor(list(unique_edges), dtype=torch.long).t()
    return edge_index

def build_magnn_style_graph():
    # Load IMDB dataset
    hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
        train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0)
        
    # Define metapaths
    metapaths = [
        [('movie', 'to_director', 'director'), ('director', 'to_movie', 'movie')],  # MDM
        [('movie', 'to_actor', 'actor'), ('actor', 'to_movie', 'movie')],          # MAM
        [('movie', 'to_keyword', 'keyword'), ('keyword', 'to_movie', 'movie')]     # MKM
    ]
    
    # Create new graph
    new_hg = pyg.data.HeteroData()
    
    # Store original edge counts
    original_edge_counts = {}
    for edge_type in hg.edge_types:
        original_edge_counts[edge_type] = hg[edge_type].edge_index.size(1)
    
    # Store node features and counts
    for node_type in hg.node_types:
        new_hg[node_type].num_nodes = hg[node_type].num_nodes
    
    # Process features
    new_hg['movie'].x = features
    
    # Calculate features for other nodes
    for node_type in ['director', 'actor', 'keyword']:
        related_edges = hg[('movie', f'to_{node_type}', node_type)].edge_index
        node_feats = scatter_mean(features[related_edges[0]], related_edges[1], 
                                dim=0, dim_size=hg[node_type].num_nodes)
        new_hg[node_type].x = node_feats
    
    # Process metapaths
    metapath_instances = {}
    
    for mp_idx, metapath in enumerate(metapaths):
        # Get ALL instances
        instances = extract_all_metapath_instances(hg, metapath)
        metapath_instances[mp_idx] = instances
        
        # Create edges from instances
        edge_indices = process_metapath_edges(instances, hg[metapath[0][0]].num_nodes)
        metapath_indices = convert_instances_to_tensors(instances)
        
        # Store in graph
        metapath_name = f"metapath_{mp_idx}"
        new_hg[metapath[0][0], metapath_name, metapath[-1][2]].edge_index = edge_indices
        new_hg[metapath[0][0], metapath_name, metapath[-1][2]].metapath_indices = metapath_indices
    
    # Store metadata
    new_hg.metapath_dict = {i: metapath for i, metapath in enumerate(metapaths)}
    new_hg.num_labels = num_labels
    new_hg.original_edge_counts = original_edge_counts
    
    # Store labels and masks
    new_hg['movie'].y = labels
    new_hg['movie'].train_mask = train_mask
    new_hg['movie'].val_mask = valid_mask
    new_hg['movie'].test_mask = test_mask
    
    # Store indices
    new_hg.train_indices = train_indices
    new_hg.valid_indices = valid_indices
    new_hg.test_indices = test_indices
    
    return new_hg, metapath_instances


def analyze_metapath_instances(instances, metapath, prefix="", hg=None):
    """Analyze metapath instances for patterns like self-loops."""
    if not instances:
        return
    
    # Count metapath self-loops (start = end node)
    metapath_self_loops = sum(1 for seq in instances if seq[0] == seq[-1])
    
    # Check for direct self-loops in original graph
    direct_self_loops = 0
    if hg is not None:
        for src_type, edge_type, dst_type in metapath:
            edge_index = hg[src_type, edge_type, dst_type].edge_index
            direct_self_loops += sum(1 for i in range(edge_index.size(1)) 
                                   if edge_index[0, i] == edge_index[1, i])
    
    # Count unique nodes at each position
    pos_nodes = defaultdict(set)
    for seq in instances:
        for i, node in enumerate(seq):
            pos_nodes[i].add(node)
            
    # Count node reuse within paths (excluding self-loops)
    reused_nodes = 0
    for seq in instances:
        if len(set(seq)) < len(seq) and seq[0] != seq[-1]:
            reused_nodes += 1
    
    print(f"{prefix}Path Analysis:")
    print(f"{prefix}  Total instances: {len(instances):,}")
    print(f"{prefix}  Metapath self-loops (start=end): {metapath_self_loops:,} ({metapath_self_loops/len(instances)*100:.2f}%)")
    print(f"{prefix}  Direct self-loops in original edges: {direct_self_loops}")
    print(f"{prefix}  Paths with reused nodes (excluding self-loops): {reused_nodes:,}")
    print(f"{prefix}  Unique nodes per position:")
    for pos, nodes in pos_nodes.items():
        print(f"{prefix}    Position {pos}: {len(nodes):,} unique nodes")
    
    if len(instances) > 0:
        print(f"{prefix}  Example instance: {instances[0]}")

def validate_graph(hg, metapath_instances):
    """Comprehensive validation of graph structure and metapath instances."""
    print("\nGraph Validation and Statistics:")
    print("=" * 50)
    
    # Original edge counts
    print("\nOriginal Edge Counts:")
    total_original = 0
    for edge_type, count in hg.original_edge_counts.items():
        print(f"  {edge_type}: {count:,}")
        total_original += count
    print(f"Total original edges: {total_original:,}")
    
    # Node information
    print("\nNode Statistics:")
    for node_type in hg.node_types:
        print(f"\n{node_type}:")
        print(f"  Nodes: {hg[node_type].num_nodes:,}")
        if hasattr(hg[node_type], 'x'):
            print(f"  Feature shape: {hg[node_type].x.shape}")
    
    # Metapath information
    print("\nMetapath Statistics:")
    total_instances = 0
    total_edges = 0
    
    for mp_idx, instances in metapath_instances.items():
        metapath = hg.metapath_dict[mp_idx]
        path_name = ' -> '.join(f"{m[0]}-{m[2]}" for m in metapath)
        print(f"\nMetapath {mp_idx} ({path_name}):")
        
        # Analyze instances
        analyze_metapath_instances(instances, metapath, prefix="  ")
        
        # Edge statistics
        edge_type = ('movie', f'metapath_{mp_idx}', 'movie')
        edge_count = hg[edge_type].edge_index.size(1)
        total_edges += edge_count
        print(f"  Unique edges in graph: {edge_count:,}")
        print(f"  Edge reduction ratio: {len(instances)/edge_count:.2f}")
        
        total_instances += len(instances)
    
    print("\nOverall Summary:")
    print(f"Total metapath instances: {total_instances:,}")
    print(f"Total unique edges: {total_edges:,}")
    print(f"Average number of instances per edge: {total_instances/total_edges:.2f}")

# Define model with metapath instance processing
class AutoHGNN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8):
        super().__init__()
        self.han_conv = AutoHGNNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=0.6,
            metadata=hg.metadata()
        )
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, metapath_dict):
        out = self.han_conv(x_dict, edge_index_dict, metapath_dict)
        out = self.lin(out['movie'])
        return out

if __name__ == "__main__":
    # Get graph with metapath instances
    hg, metapath_instances = build_magnn_style_graph()
    
    # Validate graph and print statistics
    validate_graph(hg, metapath_instances)
    
    # Model initialization
    in_channels = {node_type: hg[node_type].x.size(1) for node_type in hg.node_types}
    model = AutoHGNN(in_channels=in_channels, out_channels=hg.num_labels)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    hg, model = hg.to(device), model.to(device)
    
    # Check requires_grad for parameters
    print("\nChecking requires_grad status of model parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    # Get metapath indices dict for model
    metapath_dict = {f"metapath_{i}": hg[('movie', f"metapath_{i}", 'movie')].metapath_indices 
                     for i in range(len(hg.metapath_dict))}
    
    # Initialize lazy modules
    with torch.no_grad():
        out = model(hg.x_dict, hg.edge_index_dict, metapath_dict)
    
    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    # Training loop with early stopping
    best_val_micro_f1 = 0
    best_val_macro_f1 = 0
    best_state = None
    start_patience = patience = 100

    print("\nStarting training...")
    for epoch in range(1, 200):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(hg.x_dict, hg.edge_index_dict, metapath_dict)
        mask = hg['movie'].train_mask
        loss = F.binary_cross_entropy_with_logits(out[mask], hg['movie'].y[mask])
        loss.backward()
        optimizer.step()

            # Print gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Total gradient norm: {total_norm:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            logits = model(hg.x_dict, hg.edge_index_dict, metapath_dict)
            pred_probs = torch.sigmoid(logits)
            pred = (pred_probs > 0.5).float()

            # Handle cases where no predictions are made
            no_pred_mask = pred.sum(dim=1) == 0
            max_indices = torch.argmax(pred_probs, dim=1)
            pred[no_pred_mask, :] = 0
            pred[no_pred_mask, max_indices[no_pred_mask]] = 1

            # Evaluate all splits
            metrics = []
            for split in ['train_mask', 'val_mask', 'test_mask']:
                mask = hg['movie'][split]
                y_true = hg['movie'].y[mask]
                y_pred = pred[mask]
                
                true_pos = (y_true * y_pred).sum(dim=0)
                pred_pos = y_pred.sum(dim=0)
                actual_pos = y_true.sum(dim=0)
                
                micro_precision = true_pos.sum() / (pred_pos.sum() + 1e-8)
                micro_recall = true_pos.sum() / (actual_pos.sum() + 1e-8)
                micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
                
                precision = true_pos / (pred_pos + 1e-8)
                recall = true_pos / (actual_pos + 1e-8)
                f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
                macro_f1 = f1_per_class[~torch.isnan(f1_per_class)].mean()
                
                metrics.append({
                    'micro_f1': float(micro_f1),
                    'macro_f1': float(macro_f1)
                })

        # Print epoch results
        print(f'\nEpoch: {epoch:03d}, Loss: {loss:.4f}')
        print(f'Micro F1 -> Train: {metrics[0]["micro_f1"]:.4f}, Val: {metrics[1]["micro_f1"]:.4f}, Test: {metrics[2]["micro_f1"]:.4f}')
        print(f'Macro F1 -> Train: {metrics[0]["macro_f1"]:.4f}, Val: {metrics[1]["macro_f1"]:.4f}, Test: {metrics[2]["macro_f1"]:.4f}')
        
        # Early stopping
        val_macro_f1 = metrics[1]['macro_f1']
        if best_val_macro_f1 <= val_macro_f1:
            patience = start_patience
            best_val_macro_f1 = val_macro_f1
            best_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metrics': metrics
            }
        else:
            patience -= 1
            if patience <= 0:
                print(f'\nEarly stopping after {epoch} epochs!')
                print(f'Best validation macro F1: {best_val_macro_f1:.4f}')
                print(f'Corresponding test macro F1: {best_state["metrics"][2]["macro_f1"]:.4f}')
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])