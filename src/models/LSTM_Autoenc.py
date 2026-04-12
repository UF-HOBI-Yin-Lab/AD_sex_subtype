import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Autoencoder(nn.Module):
    def __init__(self, tf_dim, fea_dim, layers, dp_rate):
        super(LSTM_Autoencoder, self).__init__()
        
        self.tf_dim = tf_dim
        self.fea_dim = fea_dim
        self.layers = layers
        
        # Encoder
        self.lstm1 = nn.LSTM(input_size=fea_dim, hidden_size=layers[0], batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        self.lstm2 = nn.LSTM(input_size=layers[0], hidden_size=layers[1], batch_first=True)
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        self.lstm3 = nn.LSTM(input_size=layers[1], hidden_size=layers[2], batch_first=True)
        self.batch_norm3 = nn.BatchNorm1d(layers[2])
        self.lstm4 = nn.LSTM(input_size=layers[2], hidden_size=layers[3], batch_first=True)
        self.batch_norm4 = nn.BatchNorm1d(layers[3])
        
        # Binary output
        self.binary_output = nn.Linear(layers[3], layers[4])
        
        # Decoder
        self.lstm5 = nn.LSTM(input_size=layers[3], hidden_size=layers[2], batch_first=True)
        self.batch_norm5 = nn.BatchNorm1d(layers[2])
        self.lstm6 = nn.LSTM(input_size=layers[2], hidden_size=layers[1], batch_first=True)
        self.batch_norm6 = nn.BatchNorm1d(layers[1])
        self.lstm7 = nn.LSTM(input_size=layers[1], hidden_size=layers[0], batch_first=True)
        self.batch_norm7 = nn.BatchNorm1d(layers[0])
        
        # Reconstruction output
        self.reconstruction_output = nn.Linear(layers[0], fea_dim)
        self.dropout = nn.Dropout(p=dp_rate)

    def forward(self, x):
        # Encoder
        x, _ = self.lstm1(x)
        x = self.batch_norm1(x.permute(0, 2, 1)) 
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x.permute(0, 2, 1))
        x = self.batch_norm2(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x.permute(0, 2, 1))
        x = self.batch_norm3(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm4(x.permute(0, 2, 1))
        x = self.batch_norm4(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        # Latent representation
        latent = x[:, -1, :]
        # Binary output
        binary_out = self.binary_output(latent)

        # Decoder
        x, _ = self.lstm5(x)
        x = self.batch_norm5(x.permute(0, 2, 1)) 
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm6(x.permute(0, 2, 1))
        x = self.batch_norm6(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm7(x.permute(0, 2, 1))
        x = self.batch_norm7(x.permute(0, 2, 1))
        x = F.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        
        # reconstruction output
        reconstruction_out = self.reconstruction_output(x)

        return binary_out, reconstruction_out

