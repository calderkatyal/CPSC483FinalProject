import matplotlib.pyplot as plt
import torch
import numpy as np
from main import *  # This imports all the necessary components from your main script

def train_epoch(model, optimizer):
    """Modified training function that takes model and optimizer as parameters"""
    model.train()
    optimizer.zero_grad()
    out = model(hg.x_dict, hg.edge_index_dict)
    mask = hg['movie'].train_mask
    loss = F.binary_cross_entropy_with_logits(out[mask], hg['movie'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def test_epoch(model):
    """Modified test function that takes model as parameter"""
    model.eval()
    with torch.no_grad():
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
            
            true_pos = (y_true * y_pred).sum(dim=0)
            pred_pos = y_pred.sum(dim=0)
            actual_pos = y_true.sum(dim=0)
            
            precision = true_pos / (pred_pos + 1e-8)
            recall = true_pos / (actual_pos + 1e-8)
            f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)
            macro_f1 = f1_per_class[~torch.isnan(f1_per_class)].mean()
            
            metrics.append(float(macro_f1))
            
        return metrics  # [train_f1, val_f1, test_f1]

def run_model_and_collect_metrics(model_type, metapath_encoder=None):
    # Reset seed for consistency
    set_seed(483)
    
    # Initialize model based on type
    if model_type == 'han':
        model = HAN(in_channels={'movie': features.shape[1]}, 
                   out_channels=num_labels).to(device)
    else:  # HANME
        in_channels = {node_type: node_features[node_type].shape[1] 
                      for node_type in node_features}
        model = HANME(in_channels=in_channels, 
                     out_channels=num_labels,
                     metapath_encoder=metapath_encoder).to(device)
    
    # Initialize lazy modules
    with torch.no_grad():
        out = model(hg.x_dict, hg.edge_index_dict)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    
    train_metrics = []
    val_metrics = []
    test_metrics = []
    
    for epoch in range(1, 201):  # 200 epochs
        # Training
        loss = train_epoch(model, optimizer)
        
        # Testing
        metrics = test_epoch(model)
        train_metrics.append(metrics[0])
        val_metrics.append(metrics[1])
        test_metrics.append(metrics[2])
        
        if epoch % 10 == 0:  # Print progress every 10 epochs
            print(f'Epoch {epoch}: Train F1: {metrics[0]:.4f}, '
                  f'Val F1: {metrics[1]:.4f}, '
                  f'Test F1: {metrics[2]:.4f}')
    
    return train_metrics, val_metrics, test_metrics

def create_and_save_plot(metrics_dict, metric_type, filename):
    plt.figure(figsize=(10, 6))
    
    for model_name, metrics in metrics_dict.items():
        plt.plot(range(1, len(metrics) + 1), metrics, label=model_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title(f'{metric_type} Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    print("Starting model training and evaluation...")
    
    # Dictionary to store metrics for each model configuration
    models_metrics = {
        'HANME-mean': run_model_and_collect_metrics('HANME', 'mean'),
        'HANME-multihop': run_model_and_collect_metrics('HANME', 'multihop'),
        'HANME-direct': run_model_and_collect_metrics('HANME', 'direct')
    }
    
    print("All models trained. Creating plots...")
    
    # Separate metrics by type (train, val, test)
    train_metrics = {name: metrics[0] for name, metrics in models_metrics.items()}
    val_metrics = {name: metrics[1] for name, metrics in models_metrics.items()}
    test_metrics = {name: metrics[2] for name, metrics in models_metrics.items()}
    
    # Create and save plots
    create_and_save_plot(train_metrics, 'Training', 'training_performance.png')
    create_and_save_plot(val_metrics, 'Validation', 'validation_performance.png')
    create_and_save_plot(test_metrics, 'Test', 'test_performance.png')
    
    print("Plots saved successfully!")

if __name__ == '__main__':
    main()