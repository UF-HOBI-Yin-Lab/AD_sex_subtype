import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_Autoencoder(nn.Module):
    def __init__(self, tf_dim, fea_dim, layers, dp_rate):
        super(GRU_Autoencoder, self).__init__()
        
        # Encoder
        self.gru1 = nn.GRU(input_size=fea_dim, hidden_size=layers[0], batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        self.gru2 = nn.GRU(input_size=layers[0], hidden_size=layers[1], batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        self.gru3 = nn.GRU(input_size=layers[1], hidden_size=layers[2], batch_first=True)
        self.batch_norm3 = nn.BatchNorm1d(layers[2])
        
        # Binary output
        self.binary_output = nn.Linear(layers[2], layers[3])
        
        # Decoder
        self.gru4 = nn.GRU(input_size=layers[2], hidden_size=layers[1], batch_first=True)
        self.batch_norm4 = nn.BatchNorm1d(layers[1])
        self.gru5 = nn.GRU(input_size=layers[1], hidden_size=layers[0], batch_first=True)
        self.batch_norm5 = nn.BatchNorm1d(layers[0])
        
        # Reconstruction output
        self.reconstruction_output = nn.Linear(layers[0], fea_dim)
        # Dropout
        self.dropout = nn.Dropout(p=dp_rate)

        
    def forward(self, x):
        
        # Encoder
        x, _ = self.gru1(x)
        x = self.batch_norm1(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        
        x, _ = self.gru2(x.permute(0, 2, 1))
        x = self.batch_norm2(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)

        x, _ = self.gru3(x.permute(0, 2, 1))
        x = self.batch_norm3(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)
        
        # Latent representation
        latent = x[:, -1, :]
        
        # Binary output
        binary_out = self.binary_output(latent)

        # Decoder
        x, _ = self.gru4(x)
        x = self.batch_norm4(x.permute(0, 2, 1)) 
        x = F.relu(x)
        x = self.dropout(x)
        
        x, _ = self.gru5(x.permute(0, 2, 1))
        x = self.batch_norm5(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)
        
        # Reconstruction output
        reconstruction_out = self.reconstruction_output(x)

        return binary_out, reconstruction_out