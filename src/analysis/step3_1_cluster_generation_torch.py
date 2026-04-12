"""
Step 3: Cluster Analysis using trained LSTM Autoencoder
Load the trained model and extract latent representations for clustering
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==================== Model Selection ====================
from models.LSTM_Autoenc import LSTM_Autoencoder
from utils.config_LSTM_Autoenc import config
from project_paths import STEP1_3D_NPZ, STEP1_SUBSEQ_NPZ, STEP3_LATENT_NPZ, STEP3_CLUSTER_CSV, PCA_FIG_PATH
# ================================================

def load_trained_model(checkpoint_path, tf_dim, fea_dim, layers, dropout, device):
    """
    Load a trained LSTM Autoencoder model from checkpoint
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        tf_dim: Time dimension (sequence length)
        fea_dim: Feature dimension
        layers: List of layer sizes [256, 128, 64, 32, 1]
        dropout: Dropout rate
        device: torch device (cuda or cpu)
    
    Returns:
        model: Loaded LSTM_Autoencoder model in eval mode
        checkpoint: Checkpoint dict with epochs and metrics
    """
    # Initialize model with same architecture as training
    model = LSTM_Autoencoder(tf_dim, fea_dim, layers, dropout)
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  - Trained for {checkpoint['epochs']} epochs")
    if 'bestMtc' in checkpoint and checkpoint['bestMtc'] is not None:
        print(f"  - Best metrics: {checkpoint['bestMtc']}")
    
    return model, checkpoint

def extract_latent_representations(model, data_x, batch_size=64, device='cuda:0'):
    """
    Extract latent representations (encoder output) from the trained model
    
    Args:
        model: Trained LSTM_Autoencoder model
        data_x: Input data, shape (N, T, F)
        batch_size: Batch size for inference
        device: torch device
    
    Returns:
        latent_features: Latent representations, shape (N, latent_dim)
        predictions: Binary predictions, shape (N,)
        reconstructions: Reconstructed sequences, shape (N, T, F)
    """
    model.eval()
    
    n_samples = data_x.shape[0]
    latent_dim = model.layers[3]
    
    # Pre-allocate arrays
    latent_features = np.zeros((n_samples, latent_dim), dtype=np.float32)
    predictions = np.zeros(n_samples, dtype=np.float32)
    reconstructions = np.zeros_like(data_x, dtype=np.float32)
    
    # Process in batches to avoid memory issues
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, n_samples)
            batch_x = torch.FloatTensor(data_x[i:end_idx]).to(device)
            
            # Forward pass
            binary_out, rec_out = model(batch_x)
            
            x = batch_x
            for lstm_layer, bn_layer in [(model.lstm1, model.batch_norm1),
                                          (model.lstm2, model.batch_norm2),
                                          (model.lstm3, model.batch_norm3),
                                          (model.lstm4, model.batch_norm4)]:
                x, _ = lstm_layer(x)
                x = bn_layer(x.permute(0, 2, 1))
                x = torch.relu(x)
                x = model.dropout(x)
                x = x.permute(0, 2, 1)
            latent = x[:, -1, :]
            
            # Store results
            latent_features[i:end_idx] = latent.cpu().numpy()
            predictions[i:end_idx] = torch.sigmoid(binary_out).squeeze().cpu().numpy()
            reconstructions[i:end_idx] = rec_out.cpu().numpy()
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {end_idx}/{n_samples} samples...")
    
    return latent_features, predictions, reconstructions

def main():
    # 1. Load configuration
    params = config()
    print("=" * 60)
    print("LSTM Autoencoder - Cluster Analysis")
    print("=" * 60)
    print(f"Model: {params.model_name}")
    print(f"Layers: {params.layers}")
    print(f"Device: {params.device}")
    
    # 2. Load test data (3D array format)
    data_path = STEP1_3D_NPZ
    print(f"\nLoading data from: {data_path}")
    
    npz = np.load(data_path)
    data_x = npz['data_x'].astype(np.float32)  # Shape: (N, T, F)
    data_y = npz['data_y']  # Shape: (N, T, 1) or (N, 1)
    subseq_meta = np.load(STEP1_SUBSEQ_NPZ, allow_pickle=True)
    subseq_patid_list = subseq_meta['PATID']
    col_names = [str(col) for col in subseq_meta['col_name']]
    
    # Handle data_y shape
    if data_y.ndim == 3:
        data_y = data_y[:, -1, :]  # Take last timestep, shape: (N, 1)
    
    print(f"Data loaded: X shape={data_x.shape}, y shape={data_y.shape}")
    
    # 3. Build checkpoint path
    layer_info = '-'.join(map(str, params.layers))
    source_inf = '-'.join(params.data_sources)
    checkpoint_dir = params.savePath
    checkpoint_file = f"{params.model_name}_bs{params.batchSize}_lr{params.lr}_dp{params.dropout}_rdp{params.rec_dropout}_clsw{params.cls_weight}_recw{params.rec_weight}_cf{params.fold}_model.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    
    print(f"\nLooking for checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found!")
        print(f"  Please check if the model has been trained.")
        return
    
    # 4. Load trained model
    tf_dim = data_x.shape[1]  # Time dimension
    fea_dim = data_x.shape[2]  # Feature dimension
    
    model, checkpoint = load_trained_model(
        checkpoint_path, 
        tf_dim, 
        fea_dim, 
        params.layers, 
        params.dropout, 
        params.device
    )
    
    # 5. Extract latent representations
    print(f"\nExtracting latent representations...")
    latent_features, predictions, reconstructions = extract_latent_representations(
        model, 
        data_x, 
        batch_size=params.batchSize, 
        device=params.device
    )
    
    print(f"\n✓ Extraction complete!")
    print(f"  - Latent features shape: {latent_features.shape}")
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Reconstructions shape: {reconstructions.shape}")
    
    
    # 6. clustering analysis
    z1 = sch.linkage(latent_features, 'ward')
    cluster_labels = sch.cut_tree(z1, height=45)
    cluster_labels = cluster_labels.reshape(cluster_labels.size, )
    print(cluster_labels.shape, np.unique(cluster_labels))
    
    # 7. Save latent representations for clustering
    np.savez_compressed(
        STEP3_LATENT_NPZ,
        latent_features=latent_features,
        cluster_labels=cluster_labels,
        predictions=predictions,
        reconstructions=reconstructions,
        true_labels=data_y
    )
    print(f"\n✓ Latent representations saved to: {STEP3_LATENT_NPZ}")

    timestep_idx = np.array([int(str(pid).rsplit('_', 1)[-1]) for pid in subseq_patid_list])
    patient_ids = np.array([str(pid).rsplit('_', 1)[0] for pid in subseq_patid_list])
    last_step_features = data_x[np.arange(len(data_x)), timestep_idx, :]
    labels = data_y.reshape(len(data_y), -1)[:, -1].astype(int)

    df_cluster = pd.DataFrame(last_step_features, columns=col_names[1:])
    df_cluster.insert(0, 'cluster', cluster_labels.astype(int))
    df_cluster.insert(0, 'label', labels)
    df_cluster.insert(0, 'subseq_PATID', subseq_patid_list.astype(str))
    df_cluster.insert(0, 'PATID', patient_ids)
    df_cluster.insert(2, 'new_PATID', subseq_patid_list.astype(str))
    df_cluster.to_csv(STEP3_CLUSTER_CSV, index=False)
    print(f"✓ Cluster state table saved to: {STEP3_CLUSTER_CSV}")

    # 8. PCA visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(latent_features)

    principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    principal_df['Cluster'] = cluster_labels
    principal_df['Sex'] = data_y.flatten()

    # Define shape mapping: 0 → circle, 1 → square
    shapes = {0: 'o', 1: 's'}

    plt.figure(figsize=(8, 6))

    for sex in principal_df['Sex'].unique():
        subset = principal_df[principal_df['Sex'] == sex]
        scatter = plt.scatter(
            subset['Principal Component 1'],
            subset['Principal Component 2'],
            c=subset['Cluster'],
            cmap='viridis',
            alpha=0.5,
            marker=shapes[int(sex)],
            label=f'Sex {int(sex)}'
        )

    plt.title('PCA of Subtypes with Sex')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(False)

    # Add cluster color legend
    handles, labels = scatter.legend_elements(prop='colors')
    legend1 = plt.legend(handles, labels, title='Cluster', loc="upper right")
    plt.gca().add_artist(legend1)

    # Add sex shape legend
    shape_legend = [plt.Line2D([0], [0], marker=shapes[s], color='w', markerfacecolor='k', markersize=10, label=f'Sex {s}') for s in shapes]
    plt.legend(handles=shape_legend, title='Sex', loc="upper left")
    os.makedirs(PCA_FIG_PATH.parent, exist_ok=True)
    plt.savefig(PCA_FIG_PATH, dpi=300, format='svg')
    plt.show()


if __name__ == '__main__':
    main()
