import torch
import os
import numpy as np
import random
import pandas as pd
import datetime

from torch.utils.data import TensorDataset,random_split
from torch.utils.data import DataLoader, TensorDataset

from models.MLP_Autoenc import *
from utils.config_MLP_Autoenc import *
from utils.metrics import *
from project_paths import STEP1_3D_NPZ, STEP1_SUBSEQ_NPZ

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

import logging

# # Logging setup
# logging.basicConfig(
#     filename="predictions.log",  # log filename
#     level=logging.INFO,          # log level
#     format="%(asctime)s - %(message)s"  # log format
# )
class Trainer:
    def __init__(self, model, loaders, optimizer, scheduler, lossfn, rec_lossfn, params):
        self.gpu_id = params.device
        self.model = model.to(self.gpu_id)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfn = lossfn
        self.rec_lossfn = rec_lossfn
        self.params = params
        self.epochs_run = 0
        self.current_epoch = 0
        if not os.path.exists(params.savePath):  # Create save directory if needed
            os.makedirs(params.savePath, exist_ok=True)
        self.snapshot_path = f"%s{self.params.model_name}_bs{self.params.batchSize}_lr{self.params.lr}_dp{self.params.dropout}_rdp{self.params.rec_dropout}_clsw{self.params.cls_weight}_recw{self.params.rec_weight}_cf{self.params.fold}_model.pt" % params.savePath       
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        print('Load model now')
        # loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=self.gpu_id)
        self.model.load_state_dict(snapshot["model"])
        self.epochs_run = snapshot["epochs"]

    def _save_snapshot(self, epochs, bestMtc=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc, 'model': self.model.state_dict()}
        torch.save(stateDict, self.snapshot_path)
        print(f"Epoch {epochs} | Training snapshot saved at {self.snapshot_path}")
                
    def train(self):
        train_dataloader, val_dataloader, test_dataloader = self.loaders
        best_record = {'train_loss': 0, 'train_acc': 0, 'train_f': 0, 'train_auroc': 0, 'train_auprc': 0,'train_pre': 0,'train_rec': 0, 'valid_loss': 0,  'valid_acc': 0, 'valid_f': 0, 'valid_auroc': 0, 'valid_auprc': 0,  'valid_pre': 0, 'valid_rec': 0}
        nobetter, best_f1 = 0, 0.0
        
        for epoch in range(self.epochs_run, self.params.num_epochs):
            self.current_epoch = epoch
            train_loss, train_acc, train_f, train_auroc, train_auprc, train_pre, train_rec = self.train_epoch(train_dataloader, self.model, self.lossfn, self.rec_lossfn, self.optimizer, self.gpu_id, self.params.cls_weight, self.params.rec_weight, self.params.thres)
            val_loss, val_acc, val_f, val_auroc, val_auprc, val_pre, val_rec  = self.val_epoch(val_dataloader, self.model, self.lossfn, self.rec_lossfn, self.gpu_id, self.params.cls_weight, self.params.rec_weight, self.params.thres)
            self.scheduler.step(val_loss)
            print(
                ">>>Epoch:{} of Train Loss:{:.3f}, Valid Loss:{:.3f}\n"
                "Train Acc:{:.3f}, Train F1-score:{:.3f}, Train AUROC:{:.3f}, Train AUPRC:{:.3f}, Train Precision:{:.3f}, Train Recall:{:.3f};\n"
                "Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid AUROC:{:.3f}, Valid AUPRC:{:.3f}, Valid Precision:{:.3f}, Valid Recall:{:.3f}!!!\n".format(
                    epoch, train_loss, val_loss,
                    train_acc, train_f, train_auroc, train_auprc, train_pre, train_rec,
                    val_acc, val_f, val_auroc, val_auprc, val_pre, val_rec))
            if best_f1 < val_f:
                nobetter = 0
                best_f1 = val_f
                best_record['train_loss'] = train_loss
                best_record['val_loss'] = val_loss
                best_record['train_acc'] = train_acc
                best_record['val_acc'] = val_acc
                best_record['train_f'] = train_f
                best_record['val_f'] = val_f
                best_record['train_auroc'] = train_auroc
                best_record['val_auroc'] = val_auroc
                best_record['train_auprc'] = train_auprc
                best_record['val_auprc'] = val_auprc
                best_record['train_pre'] = train_pre
                best_record['val_pre'] = val_pre
                best_record['train_rec'] = train_rec
                best_record['val_rec'] = val_rec
                print(f'>Bingo!!! Get a better Model with valid f1: {best_f1:.3f}!!!')
                self._save_snapshot(epoch, best_f1)
            else:
                nobetter += 1
                if nobetter >= self.params.earlyStop:
                    print(f'Valid f1 has not improved for more than {self.params.earlyStop} steps in epoch {epoch}, stop training.')
                    break
        print("Finally,the model's Valid Acc:{:.3f}, Valid F1-score:{:.3f}, Valid AUROC:{:.3f}, Valid AUPRC:{:.3f}, Valid Precision:{:.3f}, Valid Recall:{:.3f}!!!\n\n\n".format(
                best_record['val_acc'], best_record['val_f'],  best_record['val_auroc'], best_record['val_auprc'], best_record['val_pre'], best_record['val_rec']))            
        self.test(test_dataloader, self.model, self.gpu_id, threshold=self.params.thres)
        
    def train_epoch(self, train_dataloader, model, loss_fn, rec_loss_fn, optimizer, gpu_id, cls_weight, rec_weight, threshold=0.5):
        train_loss, train_acc, train_f, train_auroc, train_auprc, train_pre, train_rec = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        model.train()
        pred_list, prob_list, label_list = [], [], []
        epoch_loss = 0.0
        num_batches = len(train_dataloader)
        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
            optimizer.zero_grad()
            pred, rec_pred = model(x_batch)
            # print('pred', pred.shape, 'rec_pred', rec_pred.shape, 'x_batch', x_batch.shape)
            cls_loss = loss_fn(pred, y_batch)
            rec_loss = rec_loss_fn(rec_pred, x_batch)
            loss = cls_weight * cls_loss + rec_weight * rec_loss # transformers: 0.5, 0.5, others: 0.8, 0.2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            pred_label = (pred > threshold).float()
            prob_list.extend(pred.detach().cpu().numpy())
            pred_list.extend(pred_label.detach().cpu().numpy())
            label_list.extend(y_batch.cpu().numpy())
                
        train_loss = epoch_loss/num_batches
        train_acc, train_f, train_auroc, train_auprc, train_pre, train_rec = accuracy(label_list, pred_list), f1(label_list, pred_list), auroc(label_list, prob_list), auprc(label_list, pred_list), precision(label_list, pred_list), recall(label_list, pred_list)
        return train_loss, train_acc, train_f, train_auroc, train_auprc, train_pre, train_rec
    
    def val_epoch(self, val_dataloader, model, loss_fn, rec_loss_fn, gpu_id, cls_weight, rec_weight, threshold=0.5):
        val_loss, val_acc, val_f, val_auroc, val_auprc, val_pre, val_rec  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        model.eval()
        pred_list, prob_list, label_list = [], [], []
        epoch_loss = 0.0
        num_batches = len(val_dataloader)
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(val_dataloader):
                x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
                pred, rec_pred = model(x_batch)
                cls_loss = loss_fn(pred, y_batch)
                rec_loss = rec_loss_fn(rec_pred, x_batch)
                loss = cls_weight * cls_loss + rec_weight * rec_loss # transformers: 0.5, 0.5, others: 0.8, 0.2
                epoch_loss += loss.item()
                pred_label = (pred > threshold).float()
                prob_list.extend(pred.detach().cpu().numpy())
                pred_list.extend(pred_label.detach().cpu().numpy())
                label_list.extend(y_batch.cpu().numpy())
                # if i == num_batches-1:
                #     logging.info(f"probability: {prob_list}, gt_label: {label_list}")
        val_loss = epoch_loss/num_batches
        val_acc, val_f, val_auroc, val_auprc, val_pre, val_rec = accuracy(label_list, pred_list), f1(label_list, pred_list), auroc(label_list, prob_list), auprc(label_list, pred_list), precision(label_list, pred_list), recall(label_list, pred_list)
        return val_loss, val_acc, val_f, val_auroc, val_auprc, val_pre, val_rec  
        
    def test(self, test_dataloader, model, gpu_id, threshold=0.5):
        model.eval()
        pred_list, prob_list, label_list = [], [], [] 
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(test_dataloader):
                x_batch, y_batch = x_batch.to(gpu_id), y_batch.to(gpu_id)
                pred, _ = model(x_batch)
                pred_label = (pred > threshold).float()
                prob_list.extend(pred.detach().cpu().numpy())
                pred_list.extend(pred_label.detach().cpu().numpy())
                label_list.extend(y_batch.cpu().numpy())
        test_acc, test_f, test_auroc, test_auprc, test_pre, test_rec = accuracy(label_list, pred_list), f1(label_list, pred_list), auroc(label_list, prob_list), auprc(label_list, pred_list), precision(label_list, pred_list), recall(label_list, pred_list)
        print("The overall performance on the test data are Acc:{:.3f}, F1-score:{:.3f}, AUROC:{:.3f}, AUPRC:{:.3f}, Precision:{:.3f}, Recall:{:.3f}!!!".format(test_acc, test_f, test_auroc, test_auprc, test_pre, test_rec))

def get_dataset(data_sources, month, seed, train_ratio=0.7, test_ratio=0.2,
                data_path=str(STEP1_3D_NPZ),
                patid_path=str(STEP1_SUBSEQ_NPZ)):
    npz = np.load(data_path)
    data_X = npz['data_x']
    data_y = npz['data_y']
    data_y = data_y[:, -1, :]
    print('data_X', data_X.shape, 'data_y', data_y.shape)

    patid_data = np.load(patid_path, allow_pickle=True)
    patid_list = patid_data['PATID']
    patient_of_subseq = np.array([str(p).rsplit('_', 1)[0] for p in patid_list])

    unique_patients = np.unique(patient_of_subseq)
    np.random.seed(seed)
    np.random.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(train_ratio * n)
    n_test = int(test_ratio * n)
    n_valid = n - n_train - n_test

    train_pats = unique_patients[:n_train]
    valid_pats = unique_patients[n_train:n_train + n_valid]
    test_pats = unique_patients[n_train + n_valid:]

    train_mask = np.isin(patient_of_subseq, train_pats)
    valid_mask = np.isin(patient_of_subseq, valid_pats)
    test_mask = np.isin(patient_of_subseq, test_pats)

    X_train, y_train = data_X[train_mask], data_y[train_mask]
    X_valid, y_valid = data_X[valid_mask], data_y[valid_mask]
    X_test, y_test = data_X[test_mask], data_y[test_mask]

    tf_dim, fea_dim = data_X.shape[1], data_X.shape[2]
    print(f'Patient-level split: '
          f'{len(train_pats)} train patients ({train_mask.sum()} subseqs), '
          f'{len(valid_pats)} valid patients ({valid_mask.sum()} subseqs), '
          f'{len(test_pats)} test patients ({test_mask.sum()} subseqs)')
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDataset objects for training and testing data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, valid_dataset, test_dataset, tf_dim, fea_dim

def get_dataloader(params, train_ratio=0.7, test_ratio=0.2):
    train_dataset, valid_dataset, test_dataset, tf_dim, fea_dim = get_dataset(
        params.data_sources, params.month, params.seed,
        train_ratio=train_ratio, test_ratio=test_ratio,
        data_path=params.data_path, patid_path=params.patid_path)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=params.batchSize, shuffle=True, drop_last=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=params.batchSize, shuffle=False, drop_last=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=params.batchSize, shuffle=False, drop_last=False, num_workers=1)

    # Print data stats
    print(f"Training data: {len(train_dataset)}, Validing data: {len(valid_dataset)}, Testing data: {len(test_dataset)}")
    check_lbl('Training', train_loader)
    check_lbl('Validing', valid_loader)
    check_lbl('Testing', test_loader)

    return [train_loader, valid_loader, test_loader], tf_dim, fea_dim

def check_lbl(name, dataloader):
    zero, one = 0, 0
    for _, (_, labels) in enumerate(dataloader):
        one += torch.sum(labels == 1).item()
        zero += torch.sum(labels == 0).item()
    print(f'In {name}, there are {one} 1s, and {zero} 0s.')

def load_model_objs(tf_dim, fea_dim, params):
    layers, dp_rate, device = params.layers, params.dropout, params.device
    model = MLP_Autoencoder(tf_dim, fea_dim, layers, dp_rate)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    lossfn = nn.BCEWithLogitsLoss()
    rec_lossfn = nn.MSELoss()
    return model, optimizer, scheduler, lossfn, rec_lossfn

    
def main():
    print("Running Python File:", os.path.basename(__file__))
    params = config()
    setup_seed(params.seed)    
    starttime = datetime.datetime.now()
    data_time1 = datetime.datetime.now()
    loaders, tf_dim, fea_dim = get_dataloader(params)
    data_time2 = datetime.datetime.now()
    print(f'Data Loading Time is {(data_time2 - data_time1).seconds}s. ')
    print(f'Model name: {params.model_name}, data sources: {params.data_sources}, month: {params.month}, batch: {params.batchSize}, epoch:{params.num_epochs}, lr: {params.lr}, layers: {params.layers}, cls_weight: {params.cls_weight}, rec_weight:{params.rec_weight}, drop: {params.dropout}, rec_drop: {params.rec_dropout}')
    train_time1 = datetime.datetime.now()
    model, optimizer, scheduler, lossfn, rec_lossfn = load_model_objs(tf_dim, fea_dim, params)
    trainer = Trainer(model, loaders, optimizer, scheduler, lossfn, rec_lossfn, params)    
    trainer.train()
    train_time2= datetime.datetime.now()
    print(f'Train time is {(train_time2 - train_time1).seconds}s. ')
    endtime = datetime.datetime.now()
    print(f'Total running time of all codes is {(endtime - starttime).seconds}s. ')
    
if __name__ == '__main__':
    main()
