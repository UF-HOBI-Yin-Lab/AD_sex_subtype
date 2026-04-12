import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Autoencoder(nn.Module):
    def __init__(self, tf_dim, fea_dim, layers, dp_rate):
        super(MLP_Autoencoder, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(fea_dim, layers[0])
        self.batch_norm1 = nn.BatchNorm1d(layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.batch_norm2 = nn.BatchNorm1d(layers[1])
        self.fc3 = nn.Linear(layers[1], layers[2])
        self.batch_norm3 = nn.BatchNorm1d(layers[2])

        # Attention Layer for Latent Representation
        self.attention = nn.MultiheadAttention(embed_dim=layers[2], num_heads=4)

        # Binary output
        self.binary_output = nn.Linear(layers[2], layers[3])

        # Decoder with residual connections
        self.fc4 = nn.Linear(layers[2], layers[1])
        self.batch_norm4 = nn.BatchNorm1d(layers[1])
        self.fc5 = nn.Linear(layers[1], layers[0])
        self.batch_norm5 = nn.BatchNorm1d(layers[0])
        self.reconstruction_output = nn.Linear(layers[0], fea_dim)

        self.dropout = nn.Dropout(p=dp_rate)

    def forward(self, x):
        # Input shape: (batch_size, tf_dim, fea_dim)
        batch_size, tf_dim, fea_dim = x.size()

        # Reshape for Linear layers
        x = x.view(-1, fea_dim)  # (batch_size * tf_dim, fea_dim)

        # Encoder
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)

        # Restore temporal shape
        x = x.view(batch_size, tf_dim, -1)  # (batch_size, tf_dim, layers[2])

        # Latent representation
        # latent = x.mean(dim=1)  # (batch_size, layers[2])
        latent = x[:, -1, :]
        # Binary output
        binary_out = self.binary_output(latent)  # (batch_size, layers[3])

        # Decoder
        x = F.relu(self.batch_norm4(self.fc4(latent)))  # (batch_size, layers[1])
        x = self.dropout(x)
        x = F.relu(self.batch_norm5(self.fc5(x)))  # (batch_size, layers[0])
        x = self.dropout(x)

        # Expand back to temporal shape
        x = x.unsqueeze(1).repeat(1, tf_dim, 1)  # (batch_size, tf_dim, layers[0])

        # Reconstruction output
        reconstruction_out = self.reconstruction_output(x)  # (batch_size, tf_dim, fea_dim)

        return binary_out, reconstruction_out
