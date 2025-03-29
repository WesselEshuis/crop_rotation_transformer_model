import torch
import torch.nn as nn
import math
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CropRotationTransformer(nn.Module):
    def __init__(self, num_crops, num_embeddings, embedding_dim, num_heads, num_layers, dropout, predict_binary=False, device=None):
        super(CropRotationTransformer, self).__init__()
        self.predict_binary = predict_binary

        # Separate embedding layers for main crops and cover crops
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move the whole model to the correct device

        # Learnable [CLS] tokens
        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_dim).to(self.device))  # For main crop
        self.cls_cover_crop_token = nn.Parameter(torch.rand(1, 1, embedding_dim).to(self.device))  # For cover crop

        self.positional_encoding = self.create_positional_encoding(
            max_seq_len=6,  # A maximum assumption for sequence length
            embedding_dim=embedding_dim,
            device=self.device,
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layers for predictions
        self.fc = nn.Linear(embedding_dim, num_crops)  # Predict the next crop
        self.fc_cover_crop = nn.Linear(embedding_dim, 1)  # Predict cover crop binary label

    def create_positional_encoding(self, max_seq_len, embedding_dim, device, num_cls_tokens=2):
        """
        Generates sinusoidal positional encodings dynamically, accounting for multiple [CLS] tokens and varying sequence lengths.
        """
        total_len = max_seq_len + num_cls_tokens  # Add the number of CLS tokens to the max sequence length
        pos_encoding = torch.zeros(total_len, embedding_dim, device=device)

        position = torch.arange(0, total_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device).float() * (-math.log(10000.0) / embedding_dim))

        # Apply sine to even indices and cosine to odd indices
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        return pos_encoding.unsqueeze(0).to(device)

    def forward(self, main_seqs, cover_seqs):

        # Embed the sequences
        main_embedded = self.embedding(main_seqs)  # Shape: [seq_len_main, embedding_dim]
        cover_embedded = self.embedding(cover_seqs)  # Shape: [seq_len_cover, embedding_dim]

        # Add positional encodings
        batch_size = main_embedded.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embedding_dim]
        main_embedded = torch.cat([cls_token, main_embedded], dim=1)  # Shape: [batch_size, sequence_length + 1, embedding_dim]
        cls_cover_crop_token = self.cls_cover_crop_token.expand(batch_size, -1, -1)  # Shape: [batch_size, 1, embedding_dim]
        cover_embedded = torch.cat([cls_cover_crop_token, cover_embedded], dim=1)

        # Add positional encoding to the embeddings
        main_embedded += self.positional_encoding[:, :main_embedded.size(1), :].to(main_embedded.device)  # Match sequence length, move to device
        main_embedded += self.positional_encoding[:, :main_embedded.size(1), :].to(main_embedded.device)
        
        # Pass through the transformer encoder
        main_embedded = self.transformer_encoder(main_embedded)  # Shape: [1, seq_len_main + 1, embedding_dim]
        cover_embedded = self.transformer_encoder(cover_embedded)  # Shape: [1, seq_len_cover + 1, embedding_dim]

        # Extract the [CLS] token outputs
        cls_output_main = main_embedded[:, 0, :]  # Shape: [1, embedding_dim]
        cls_output_cover = main_embedded[:, 0, :]  # Shape: [1, embedding_dim]

        # Pass through fully connected layers to get logits
        crop_logits = self.fc(cls_output_main)  # Shape: [batch_size, num_crops]
        cover_crop_logits = self.fc_cover_crop(cls_output_cover)  # Shape: [batch_size, 1]

        return crop_logits, cover_crop_logits
