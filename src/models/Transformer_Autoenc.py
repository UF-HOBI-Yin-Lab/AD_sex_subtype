import torch
import torch.nn as nn
import math

class Transformer_Autoencoder(nn.Module):
    def __init__(self, tf_dim, fea_dim, layers, dp_rate, nhead=8):
        super(Transformer_Autoencoder, self).__init__()
        
        self.tf_dim = tf_dim  # Maximum sequence length
        self.fea_dim = fea_dim  # Input feature dimension (e.g., 1186)
        self.embedding_dim = layers[0]  # Fixed embedding dimension for Transformer (128)

        # Input Embedding
        self.embedding = nn.Linear(fea_dim, self.embedding_dim)  # Project input features to embedding_dim (1186 -> 128)
        self.embedding_activation = nn.ReLU()

        # Positional Encoding
        self.positional_encoding = self._generate_positional_encoding(self.tf_dim, self.embedding_dim)

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=nhead,
                dropout=dp_rate,
                batch_first=True
            ),
            num_layers=len(layers) - 1  # Encoder layers
        )

        # Transformer Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=nhead,
                dropout=dp_rate,
                batch_first=True
            ),
            num_layers=len(layers) - 1  # Decoder layers
        )

        # Latent Representation Projection (128 -> 32 for classification)
        self.latent_projection = nn.Linear(self.embedding_dim, layers[-2])

        # Binary Classification Output (32 -> 1)
        self.binary_output = nn.Linear(layers[-2], layers[-1])

        # Reconstruction Output (128 -> fea_dim)
        self.reconstruction_output = nn.Linear(self.embedding_dim, fea_dim)  # Map back to original feature space (128 -> 1186)

        # Dropout
        self.dropout = nn.Dropout(p=dp_rate)

    def _generate_positional_encoding(self, seq_len, model_dim):
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(seq_len, model_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # Original input for reconstruction
        original_input = x.clone()

        # Input Embedding
        x = self.embedding(x)  # Project input to embedding_dim (1186 -> 128)
        x = self.embedding_activation(x)

        # Add Positional Encoding
        batch_size, seq_len, _ = x.size()
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        x = self.dropout(x)

        # Transformer Encoder
        memory = self.encoder(x)  # Encoder output: (batch_size, seq_len, 128)

        # Latent Representation for Binary Classification
        pooled_latent = memory.mean(dim=1)  # Mean pooling over sequence
        latent = self.latent_projection(pooled_latent)  # Reduce dimension (128 -> 32)
        binary_out = self.binary_output(latent)  # Binary classification output (32 -> 1)

        # Transformer Decoder for Reconstruction
        decoded = self.decoder(memory, memory)  # Pass memory through the decoder
        reconstruction_out = self.reconstruction_output(decoded)  # Reconstruction output (128 -> 1186)

        return binary_out, reconstruction_out
