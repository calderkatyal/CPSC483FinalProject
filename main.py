import os.path as osp
from typing import Dict, List, Union
import os
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric
import torch_geometric.transforms as T
from load_imdb import load_imdb, data_loader
import numpy as np
from torch_scatter import scatter_mean
from typing import Tuple
from MetaPathAdd import MetaPathAdd
from autohgnn_conv import AutoHGNNConv
import argparse
import random

preprocessed_data_path = 'preprocessed_hg.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the IMDB dataset
hg, features, labels, num_labels, train_indices, valid_indices, test_indices, \
    train_mask, valid_mask, test_mask, node_type_names, link_type_dic, label_names = load_imdb(feat_type=0, random_state=42)

assert (train_mask & valid_mask).sum() == 0, "Train and validation masks overlap!"
assert (train_mask & test_mask).sum() == 0, "Train and test masks overlap!"
assert (valid_mask & test_mask).sum() == 0, "Validation and test masks overlap!"

print(f"Number of training samples: {train_mask.sum()}")
print(f"Number of validation samples: {valid_mask.sum()}")
print(f"Number of test samples: {test_mask.sum()}")
print(f"Training percent: {train_mask.sum() / len(train_mask) * 100}%")
print(f"Validation percent: {valid_mask.sum() / len(valid_mask) * 100}%")
print(f"Test percent: {test_mask.sum() / len(test_mask) * 100}%")




metapaths = [
    [('movie', 'to_director', 'director'), ('director', 'to_movie', 'movie')],  # MDM
    [('movie', 'to_actor', 'actor'), ('actor', 'to_movie', 'movie')]           # MAM

]

# Print initial graph stats
print("\nBefore metapath transform:")
print(f"Node types: {hg.node_types}")
print(f"Edge types: {hg.edge_types}")
for node_type in hg.node_types:
    print(f"Number of {node_type} nodes: {hg[node_type].num_nodes}")

def assign_node_features(hg, features):
    """
    Assign features to all node types using scatter mean
    """
    # Get device from features tensor
    device = features.device
    hg = hg.to(device)
    # Assign movie features directly
    hg['movie'].x = features
    hg['movie'].y = labels.to(device)

    # Calculate features for other node types
    node_types = ['director', 'actor', 'keyword']
    for node_type in node_types:
        try:
            # Find edges connecting movies to this node type
            related_edges = hg[('movie', f'to_{node_type}', node_type)].edge_index
            
            # Calculate mean features from connected movies
            node_feats = scatter_mean(
                features[related_edges[0]], 
                related_edges[1],
                dim=0, 
                dim_size=hg[node_type].num_nodes
            )
            
            # Assign calculated features
            hg[node_type].x = node_feats
        except Exception as e:
            print(f"Could not assign features for {node_type}: {e}")
    
    return hg

if os.path.exists(preprocessed_data_path):
    print("Loading preprocessed data...")
    saved_data = torch.load(preprocessed_data_path, map_location=device)
    hg = saved_data['hg']
    metapath_data = saved_data['metapath_data']
    node_features = saved_data['features']
else:
    # Move tensors to device first
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)
    test_mask = test_mask.to(device)
    
    hg = assign_node_features(hg, features)

    node_features = {}
    # Store node features
    for node_type in hg.node_types:
        node_features[node_type] = hg[node_type].x
    # Apply metapath transform
    transform = MetaPathAdd(
        metapaths=metapaths,
        drop_orig_edge_types=True,  
        drop_unconnected_node_types=True 
    )
    hg, metapath_data = transform(hg)

    # Print transformed graph stats
    print("\nAfter metapath transform:")
    print(f"Node types: {hg.node_types}")
    print(f"Edge types: {hg.edge_types}")
    for node_type in hg.node_types:
        print(f"Number of {node_type} nodes: {hg[node_type].num_nodes}")

    # Assign masks using PyG's convention (val_mask instead of valid_mask)
    hg['movie'].train_mask = train_mask
    hg['movie'].val_mask = valid_mask  # Using valid_mask from data loader but assigning as val_mask
    hg['movie'].test_mask = test_mask

    # Store additional metadata
    hg.num_labels = num_labels
    hg.train_indices = train_indices
    hg.valid_indices = valid_indices
    hg.test_indices = test_indices
    hg.node_type_names = node_type_names
    hg.link_type_dic = link_type_dic
    hg.label_names = label_names

    # Save preprocessed data
    torch.save({'hg': hg, 'metapath_data': metapath_data, 'features': node_features}, preprocessed_data_path)

hg = hg.to(device)

# Print metapath data and feature shapes
firstpairs = {k: metapath_data[k] for k in list(metapath_data)[:5]}
print("\nPRINTING FIRST FEW PAIRS OF METAPATH DATA")
print(firstpairs)

for node_type, feat_tensor in node_features.items():
    print(f"{node_type} features shape: {feat_tensor.shape}")

class AutoHGNN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8, metapath_encoder='multihop'):
        super().__init__()
        self.autohgnn_conv = AutoHGNNConv(
            in_channels, hidden_channels, 
            heads=heads,
            dropout=0.6, 
            metadata=hg.metadata(),
            metapath_data=metapath_data,
            node_features=node_features,
            metapath_encoder=metapath_encoder,
        )
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        out = self.autohgnn_conv(x_dict, edge_index_dict)
        out = self.lin(out['movie'])
        return out
def train() -> float:
    model.train()
    optimizer.zero_grad()
    out = model(hg.x_dict, hg.edge_index_dict)
    mask = hg['movie'].train_mask
    loss = F.binary_cross_entropy_with_logits(out[mask], hg['movie'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def training_schedule(lam: float, t: int, T: int, scheduler: str) -> float:
    """
    Implementation of Algorithm 2: Strategize Training Schedule
    
    Args:
        lam: Initial proportion of nodes (λ₀)
        t: Current epoch
        T: Epoch when scheduler reaches 1.0
        scheduler: Type of scheduler ('linear', 'root', or 'geom')
    """
    if t >= T:
        return 1.0
    
    if scheduler == 'linear':
        return min(1.0, lam + (1 - lam) * t/T)
    elif scheduler == 'root':
        return min(1.0, np.sqrt(lam**2 + (1 - lam**2) * t/T))
    elif scheduler == 'geom':
        return min(1.0, 2**(np.log2(lam) - np.log2(lam) * t/T))
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")
def train_with_lts(epoch: int,
                   model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   hg: dict,
                   lam: float = 0.2,
                   T: int = 50,
                   scheduler: str = 'linear') -> float:
    model.train()
    optimizer.zero_grad()

    # Step 1: Calculate loss (reduction='none')
    out = model(hg.x_dict, hg.edge_index_dict)
    mask = hg['movie'].train_mask
    train_labels = hg['movie'].y[mask]
    train_pred = out[mask]

    loss = F.binary_cross_entropy_with_logits(
        train_pred, train_labels, reduction='none'
    )

    # Step 2: Sort loss
    sorted_losses, indices = torch.sort(loss.mean(dim=1))

    # Step 3: Strategize Training Schedule
    if epoch >= T:
        # Use all nodes after T epochs (mentioned in paper text)
        loss = loss.mean()
    else:
        # Follow Algorithm 1
        size = training_schedule(lam, epoch, T, scheduler)
        num_large_losses = int(len(loss) * size)
        idx = indices[:num_large_losses]
        loss = loss[idx].mean()

    # Step 4: Loss backward
    loss.backward()
    optimizer.step()

    return float(loss)

@torch.no_grad()
def test() -> Dict[str, List[float]]:
    model.eval()
    logits = model(hg.x_dict, hg.edge_index_dict)
    pred_probs = torch.sigmoid(logits)
    pred = (pred_probs > 0.5).float()

    # Handle cases where no predictions are made
    no_pred_mask = pred.sum(dim=1) == 0
    max_indices = torch.argmax(pred_probs, dim=1)
    pred[no_pred_mask, :] = 0
    pred[no_pred_mask, max_indices[no_pred_mask]] = 1

    metrics = []
    for split in ['train_mask', 'val_mask', 'test_mask']:  
        mask = hg['movie'][split]
        y_true = hg['movie'].y[mask]
        y_pred = pred[mask]
        
        # Calculate Micro F1
        true_pos = (y_true * y_pred).sum(dim=0)
        pred_pos = y_pred.sum(dim=0)
        actual_pos = y_true.sum(dim=0)
        
        micro_precision = true_pos.sum() / (pred_pos.sum() + 1e-8)
        micro_recall = true_pos.sum() / (actual_pos.sum() + 1e-8)
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
        
        # Calculate Macro F1
        precision = true_pos / (pred_pos + 1e-8)
        recall = true_pos / (actual_pos + 1e-8)
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
        macro_f1 = f1_per_class[~torch.isnan(f1_per_class)].mean()

        # Print example incorrect prediction for test set
        if split == 'test_mask':
            incorrect = ~(y_true == y_pred).all(dim=1)
            if incorrect.any():
                idx = torch.where(incorrect)[0][torch.randint(0, incorrect.sum(), (1,))].item()
                true_labs = [label_names[i] for i in range(len(label_names)) if y_true[idx][i].item() == 1]
                pred_labs = [label_names[i] for i in range(len(label_names)) if y_pred[idx][i].item() == 1]
                probs = pred_probs[mask][idx]
                
                print("\nRandom Incorrect Prediction:")
                print("True:", true_labs)
                print("Pred:", pred_labs)
                print("\nAll probabilities:")
                for lab, prob in zip(label_names, probs):
                    print(f"{lab}: {prob:.4f}")
        
        metrics.append({
            'micro_f1': float(micro_f1),
            'macro_f1': float(macro_f1)
        })

    print(f'Micro F1 -> Train: {metrics[0]["micro_f1"]:.4f}, Val: {metrics[1]["micro_f1"]:.4f}, Test: {metrics[2]["micro_f1"]:.4f}')
    print(f'Macro F1 -> Train: {metrics[0]["macro_f1"]:.4f}, Val: {metrics[1]["macro_f1"]:.4f}, Test: {metrics[2]["macro_f1"]:.4f}')
    
    return {'micro_f1': [m['micro_f1'] for m in metrics],
            'macro_f1': [m['macro_f1'] for m in metrics]}
def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)  # For Python hash randomization
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
    # Ensures reproducibility when using CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Main training loop
if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--with_lts', action='store_true', help='Use LTS for training')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--lam', type=float, default=0.2, help='Initial proportion of nodes')
    parser.add_argument('--T', type=int, default=50, help='Epoch to reach full dataset')
    parser.add_argument('--scheduler', type=str, default='linear', help='Type of scheduler')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--metapath_encoder', type=str, default='multihop', help='Type of metapath encoder')
    args = parser.parse_args()
    # Initialize model
    SEED=483
    set_seed(SEED)
    in_channels = {node_type: node_features[node_type].shape[1] for node_type in node_features}
    model = AutoHGNN(in_channels=in_channels, out_channels=num_labels, metapath_encoder = args. metapath_encoder)
    model = model.to(device)

    # Initialize lazy modules
    with torch.no_grad():
        out = model(hg.x_dict, hg.edge_index_dict)

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    # Initialize LTS parameters
    lam = args.lam       # Initial proportion of nodes
    T = args.T           # Epoch when scheduler reaches 1.0
    scheduler = args.scheduler  # Type of scheduler

    # Training loop with LTS
    best_val_micro_f1 = 0
    best_val_macro_f1 = 0
    best_state = None
    start_patience = patience = args.patience

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}:')
        loss = train_with_lts(epoch, model, optimizer, hg, lam, T, scheduler) if args.with_lts else train()
        print(f'Train Loss: {loss:.4f}')
        metrics = test()
        
        val_micro_f1 = metrics['micro_f1'][1]
        val_macro_f1 = metrics['macro_f1'][1]
        
        if best_val_macro_f1 <= val_macro_f1:
            patience = start_patience
            best_val_macro_f1 = val_macro_f1
            best_val_micro_f1 = val_micro_f1
            best_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_macro_f1': val_macro_f1,
                'val_micro_f1': val_micro_f1,
                'test_macro_f1': metrics['macro_f1'][2],
                'test_micro_f1': metrics['micro_f1'][2]
            }
        else:
            patience -= 1

        if patience <= 0:
            print(f'\nEarly stopping after {epoch} epochs!')
            print(f'Best validation macro F1: {best_val_macro_f1:.4f}')
            print(f'Best validation micro F1: {best_val_micro_f1:.4f}')
            print(f'Corresponding test macro F1: {best_state["test_macro_f1"]:.4f}')
            print(f'Corresponding test micro F1: {best_state["test_micro_f1"]:.4f}')
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])

    # Evaluate best model
    test_metrics = test()
    print("\nFinal test metrics:")
    print(f"Micro F1: {test_metrics['micro_f1'][2]:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1'][2]:.4f}")

    # Save best model
    torch.save(model.state_dict(), 'best_model.pth')