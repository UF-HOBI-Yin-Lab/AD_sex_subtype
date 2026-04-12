import torch as t
from project_paths import STEP1_3D_NPZ, STEP1_SUBSEQ_NPZ, model_save_dir

# Set the parameter variables and change them uniformly here
class config:
    def __init__(self):
        self.thres=0.5
        #model
        self.model_name = 'MLPAuto'#MLPAuto
        self.data_sources = ['Demo', 'Phe']# Demo, Med, Phe, Lab, Vital 
        self.layers = [16, 16, 16, 1]
        self.month = 6
        # Set random seeds
        self.seed = 1234# 42 6657 2024 123 666
        self.fold = 1# 1 2 3 4 5
        # Training parameters
        self.batchSize = 64
        self.dropout = 0.0
        self.rec_dropout = 0.0
        self.num_epochs = 80
        self.lr = 0.001
        self.weight_decay = 0.001
        self.earlyStop = 20
        self.cls_weight=0.8
        self.rec_weight=0.2
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.data_path = str(STEP1_3D_NPZ)
        self.patid_path = str(STEP1_SUBSEQ_NPZ)
        self.split_level = 'patient'
        self.savePath = str(model_save_dir(self.model_name, self.data_sources, self.month, self.layers, self.seed)) + "/"
