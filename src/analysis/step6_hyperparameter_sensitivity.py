"""
Step 6: Hyperparameter sensitivity analysis for the PyTorch LSTM pipeline.

This script evaluates robustness of the clustering / subtype solution to:
  1. Clustering hyperparameters only (no retraining)
  2. Bottleneck dimension changes (full retraining)
  3. Full model scale changes (full retraining)

All retraining uses patient-level split to avoid data leakage.
"""

import copy
import os
import sys

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import torch
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_train_lstmauto import get_dataloader, setup_seed
from models.LSTM_Autoenc import LSTM_Autoencoder
from utils.config_LSTM_Autoenc import config as BaseConfig
from project_paths import STEP1_3D_NPZ, STEP1_SUBSEQ_NPZ, STEP3_LATENT_NPZ, STEP3_CLUSTER_CSV, STEP4_SUBTYPE_CSV, STEP6_SENSITIVITY_CSV, TORCH_MODEL_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_3D_PATH = str(STEP1_3D_NPZ)
PATID_PATH = str(STEP1_SUBSEQ_NPZ)
LATENT_PATH = str(STEP3_LATENT_NPZ)
RES_CSV_PATH = str(STEP3_CLUSTER_CSV)
SUBTYPE_CSV = str(STEP4_SUBTYPE_CSV)
MODEL_BASE = str(TORCH_MODEL_DIR)
SAVE_PATH = str(STEP6_SENSITIVITY_CSV)

N_STATES = 7
N_SUBTYPES = 5


def clone_config(base_cfg, *, layers=None, model_name=None):
    cfg = copy.deepcopy(base_cfg)
    if layers is not None:
        cfg.layers = list(layers)
    if model_name is not None:
        cfg.model_name = model_name

    cfg.data_path = DATA_3D_PATH
    cfg.patid_path = PATID_PATH
    cfg.split_level = "patient"

    layer_info = "-".join(map(str, cfg.layers))
    source_inf = "-".join(cfg.data_sources)
    cfg.savePath = (
        f"{MODEL_BASE}/model_{cfg.model_name}/"
        f"source{source_inf}_month{cfg.month}_layer{layer_info}_seed{cfg.seed}/"
    )
    return cfg


def build_snapshot_path(cfg):
    return (
        f"{cfg.savePath}{cfg.model_name}_bs{cfg.batchSize}_lr{cfg.lr}_dp{cfg.dropout}"
        f"_rdp{cfg.rec_dropout}_clsw{cfg.cls_weight}_recw{cfg.rec_weight}_cf{cfg.fold}_model.pt"
    )


def extract_latent_features(model, data_x, device, batch_size=256):
    model.eval()
    latent_dim = model.layers[3]
    latent_features = np.zeros((len(data_x), latent_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(data_x), batch_size):
            end = min(start + batch_size, len(data_x))
            batch_x = torch.tensor(data_x[start:end], dtype=torch.float32, device=device)

            x, _ = model.lstm1(batch_x)
            x = model.batch_norm1(x.permute(0, 2, 1))
            x = torch.relu(x)
            x = model.dropout(x)

            x, _ = model.lstm2(x.permute(0, 2, 1))
            x = model.batch_norm2(x.permute(0, 2, 1))
            x = torch.relu(x)
            x = model.dropout(x)

            x, _ = model.lstm3(x.permute(0, 2, 1))
            x = model.batch_norm3(x.permute(0, 2, 1))
            x = torch.relu(x)
            x = model.dropout(x)

            x, _ = model.lstm4(x.permute(0, 2, 1))
            x = model.batch_norm4(x.permute(0, 2, 1))
            x = torch.relu(x)
            x = model.dropout(x)
            x = x.permute(0, 2, 1)

            latent = x[:, -1, :]
            latent_features[start:end] = latent.cpu().numpy()

            if (start // batch_size) % 10 == 0:
                print(f"    {end}/{len(data_x)} windows processed...")

    return latent_features


def recluster(latent, linkage_method="ward", metric="euclidean", n_clusters=N_STATES):
    if linkage_method == "ward":
        z = sch.linkage(latent, method="ward")
    else:
        z = sch.linkage(pdist(latent, metric=metric), method=linkage_method)
    return sch.fcluster(z, n_clusters, criterion="maxclust") - 1


def get_patient_subtype_ids(df_res_new, n_top=N_SUBTYPES):
    cls_list = []
    pat_list = []

    for patid in df_res_new["PATID"].unique():
        tmp = df_res_new[df_res_new["PATID"] == patid]
        cls_list.append(str(tmp["cluster"].unique()))
        pat_list.append(patid)

    df_sub = pd.DataFrame({"PATID": pat_list, "cls_pattern": cls_list})
    top_patterns = df_sub["cls_pattern"].value_counts().head(n_top).index.tolist()
    df_sub["subtype_id"] = df_sub["cls_pattern"].map(
        {pattern: i for i, pattern in enumerate(top_patterns)}
    ).fillna(-1).astype(int)
    return df_sub[["PATID", "subtype_id"]]


def compute_metrics(base_state, new_state, df_base_sub, df_new_sub):
    state_ami = adjusted_mutual_info_score(base_state, new_state)
    state_ari = adjusted_rand_score(base_state, new_state)

    merged = df_base_sub.merge(df_new_sub, on="PATID", suffixes=("_base", "_new"))
    valid = merged[(merged["subtype_id_base"] >= 0) & (merged["subtype_id_new"] >= 0)]

    if len(valid) < 10:
        return state_ami, state_ari, np.nan, np.nan

    subtype_ami = adjusted_mutual_info_score(valid["subtype_id_base"], valid["subtype_id_new"])
    subtype_ari = adjusted_rand_score(valid["subtype_id_base"], valid["subtype_id_new"])
    return state_ami, state_ari, subtype_ami, subtype_ari


def fmt(value):
    return round(float(value), 3) if not np.isnan(value) else np.nan


def train_one_model(cfg):
    setup_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else torch.device("cpu")
    cfg.device = device

    loaders, tf_dim, fea_dim, _ = get_dataloader(cfg)
    train_loader, val_loader, _ = loaders

    model = LSTM_Autoencoder(tf_dim, fea_dim, cfg.layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    cls_lossfn = torch.nn.BCEWithLogitsLoss()
    rec_lossfn = torch.nn.MSELoss()

    snapshot_path = build_snapshot_path(cfg)
    os.makedirs(cfg.savePath, exist_ok=True)

    best_val_f1 = -1.0
    patience = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred, rec_pred = model(x_batch)
            cls_loss = cls_lossfn(pred, y_batch)
            rec_loss = rec_lossfn(rec_pred, x_batch)
            loss = cfg.cls_weight * cls_loss + cfg.rec_weight * rec_loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total = 0.0
        pred_list, label_list = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pred, rec_pred = model(x_batch)
                cls_loss = cls_lossfn(pred, y_batch)
                rec_loss = rec_lossfn(rec_pred, x_batch)
                loss = cfg.cls_weight * cls_loss + cfg.rec_weight * rec_loss
                val_loss_total += loss.item()

                pred_label = (pred > cfg.thres).float()
                pred_list.extend(pred_label.cpu().numpy())
                label_list.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(np.array(label_list).ravel(), np.array(pred_list).ravel(), zero_division=0)
        val_loss = val_loss_total / max(len(val_loader), 1)
        scheduler.step(val_loss)

        print(f"    Epoch {epoch + 1}/{cfg.num_epochs} - val_loss={val_loss:.4f} pseudo_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save(
                {"epochs": epoch, "bestMtc": best_val_f1, "model": model.state_dict()},
                snapshot_path,
            )
        else:
            patience += 1
            if patience >= cfg.earlyStop:
                print("    Early stop triggered.")
                break

    return snapshot_path


def load_or_train_and_extract(cfg, data_x):
    snapshot_path = build_snapshot_path(cfg)
    if not os.path.exists(snapshot_path):
        print("    Checkpoint not found. Training now...")
        snapshot_path = train_one_model(cfg)
    else:
        print("    Checkpoint found. Skipping training.")

    device = cfg.device if torch.cuda.is_available() else torch.device("cpu")
    cfg.device = device

    model = LSTM_Autoencoder(
        tf_dim=data_x.shape[1],
        fea_dim=data_x.shape[2],
        layers=cfg.layers,
        dp_rate=cfg.dropout,
    ).to(device)

    snapshot = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(snapshot["model"])
    print(f"    Loaded checkpoint: {snapshot_path}")
    return extract_latent_features(model, data_x, device)


def main():
    base_cfg = BaseConfig()
    base_cfg = clone_config(base_cfg, layers=base_cfg.layers, model_name=base_cfg.model_name)

    print("=" * 65)
    print("Loading baseline data...")
    print("=" * 65)

    latent_npz = np.load(LATENT_PATH)
    baseline_latent = latent_npz["latent_features"]
    latent_cluster_labels = latent_npz["cluster_labels"]

    data_3d = np.load(DATA_3D_PATH)
    data_x = data_3d["data_x"].astype(np.float32)

    df_res = pd.read_csv(RES_CSV_PATH)
    df_sub = pd.read_csv(SUBTYPE_CSV)

    if "cluster" in df_res.columns:
        baseline_state_labels = df_res["cluster"].to_numpy()
    else:
        baseline_state_labels = latent_cluster_labels

    if len(baseline_state_labels) == len(latent_cluster_labels) and np.array_equal(
        baseline_state_labels, latent_cluster_labels
    ):
        print("Alignment check passed: baseline cluster labels match latent file.")
    else:
        print("Warning: baseline cluster labels do not exactly match latent file ordering.")

    subtype_patterns = df_sub["cls_pattern"].value_counts().head(N_SUBTYPES).index.tolist()
    df_sub = df_sub[df_sub["cls_pattern"].isin(subtype_patterns)].copy()
    df_sub["subtype_id"] = df_sub["cls_pattern"].map({p: i for i, p in enumerate(subtype_patterns)})
    df_base_sub = df_sub[["PATID", "subtype_id"]]

    results = []

    print("\n" + "=" * 65)
    print("EXPERIMENT A: Clustering hyperparameters (no retraining)")
    print("=" * 65)

    cluster_configs = [
        ("Baseline (Ward + Euclidean)", "ward", "euclidean"),
        ("Complete + Euclidean", "complete", "euclidean"),
        ("Average + Euclidean", "average", "euclidean"),
        ("Average + Cosine", "average", "cosine"),
        ("Average + Correlation", "average", "correlation"),
    ]

    for name, linkage_method, metric in cluster_configs:
        print(f"\n  [{name}]")
        new_state = recluster(baseline_latent, linkage_method, metric, N_STATES)
        df_res_new = df_res.copy()
        df_res_new["cluster"] = new_state
        df_new_sub = get_patient_subtype_ids(df_res_new, N_SUBTYPES)

        state_ami, state_ari, subtype_ami, subtype_ari = compute_metrics(
            baseline_state_labels, new_state, df_base_sub, df_new_sub
        )
        print(f"    State   AMI={fmt(state_ami)}  ARI={fmt(state_ari)}")
        print(f"    Subtype AMI={fmt(subtype_ami)}  ARI={fmt(subtype_ari)}")

        results.append(
            {
                "Configuration": name,
                "State AMI": fmt(state_ami),
                "State ARI": fmt(state_ari),
                "Subtype AMI": fmt(subtype_ami),
                "Subtype ARI": fmt(subtype_ari),
            }
        )

    print("\n" + "=" * 65)
    print("EXPERIMENT B: Bottleneck dimension (full retraining)")
    print("=" * 65)

    bottleneck_configs = [16, 64, 128]
    for bottleneck_dim in bottleneck_configs:
        layers = [base_cfg.layers[0], base_cfg.layers[1], base_cfg.layers[2], bottleneck_dim, 1]
        cfg = clone_config(base_cfg, layers=layers, model_name="LSTMAuto")

        print(f"\n  [Bottleneck dim = {bottleneck_dim}]")
        new_latent = load_or_train_and_extract(cfg, data_x)
        new_state = recluster(new_latent, "ward", "euclidean", N_STATES)

        df_res_new = df_res.copy()
        df_res_new["cluster"] = new_state
        df_new_sub = get_patient_subtype_ids(df_res_new, N_SUBTYPES)

        state_ami, state_ari, subtype_ami, subtype_ari = compute_metrics(
            baseline_state_labels, new_state, df_base_sub, df_new_sub
        )
        print(f"    State   AMI={fmt(state_ami)}  ARI={fmt(state_ari)}")
        print(f"    Subtype AMI={fmt(subtype_ami)}  ARI={fmt(subtype_ari)}")

        results.append(
            {
                "Configuration": f"Bottleneck dim = {bottleneck_dim}",
                "State AMI": fmt(state_ami),
                "State ARI": fmt(state_ari),
                "Subtype AMI": fmt(subtype_ami),
                "Subtype ARI": fmt(subtype_ari),
            }
        )

    print("\n" + "=" * 65)
    print("EXPERIMENT C: Full model scale [n,n,n,n,1]")
    print("=" * 65)

    full_scale_configs = [
        ("Full scale [16,16,16,16,1]", [16, 16, 16, 16, 1]),
        ("Full scale [32,32,32,32,1]", [32, 32, 32, 32, 1]),
        ("Full scale [64,64,64,64,1]", [64, 64, 64, 64, 1]),
    ]

    for name, layers in full_scale_configs:
        cfg = clone_config(base_cfg, layers=layers, model_name="LSTMAuto")
        print(f"\n  [{name}]")
        new_latent = load_or_train_and_extract(cfg, data_x)
        new_state = recluster(new_latent, "ward", "euclidean", N_STATES)

        df_res_new = df_res.copy()
        df_res_new["cluster"] = new_state
        df_new_sub = get_patient_subtype_ids(df_res_new, N_SUBTYPES)

        state_ami, state_ari, subtype_ami, subtype_ari = compute_metrics(
            baseline_state_labels, new_state, df_base_sub, df_new_sub
        )
        print(f"    State   AMI={fmt(state_ami)}  ARI={fmt(state_ari)}")
        print(f"    Subtype AMI={fmt(subtype_ami)}  ARI={fmt(subtype_ari)}")

        results.append(
            {
                "Configuration": name,
                "State AMI": fmt(state_ami),
                "State ARI": fmt(state_ari),
                "Subtype AMI": fmt(subtype_ami),
                "Subtype ARI": fmt(subtype_ari),
            }
        )

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df_out.to_csv(SAVE_PATH, index=False)

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(df_out.to_string(index=False))
    print(f"\nSaved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
