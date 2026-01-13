"""
ChemNet-Vision Neural Network Architecture
============================================
Custom model built from scratch (no pretrained weights).

Architecture follows README_V2.md specifications:
- CNN Encoder: Custom Conv layers → 512 dim
- MLP Encoder: 23 features → 128 dim
- GNN Encoder: Graph Conv → 128 dim
- Fusion: Concatenate → 256 dim projection
- LSTM Decoder: Autoregressive SMILES generation

Author: Alexandru Gabriel
Institution: POLITEHNICA București – FIIR
Discipline: Rețele Neuronale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ============================================================================
# SIMPLE NLP MODEL - For Sentence Generation and Substance Identification
# ============================================================================

class MoleculeNLPModel(nn.Module):
    """
    Simplified NLP Model for sentence generation.
    Takes a query and generates a descriptive sentence about a molecule.
    Also used to identify molecules from text.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(MoleculeNLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)
        # embeds shape: (batch_size, seq_len, embedding_dim)
        lstm_out, hidden = self.lstm(embeds, hidden)
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        logits = self.fc(lstm_out)
        # logits shape: (batch_size, seq_len, vocab_size)
        return logits, hidden


def get_nlp_model(vocab_size):
    """Factory function for the NLP model."""
    return MoleculeNLPModel(vocab_size)


# ============================================================================
# DYNAMIC MODEL SELECTION
# ============================================================================

def get_model(vocab_size, num_node_features=9, num_numeric_features=23, mode='full'):
    """
    Returns the appropriate model based on mode.
    Mode 'nlp' is the simplified sentence generator.
    Mode 'full' is the original multimodal model.
    """
    if mode == 'nlp':
        return get_nlp_model(vocab_size)
    else:
        # Original multimodal model (kept for backward compatibility or future use)
        return ChemNetVisionModel(vocab_size, num_node_features, num_numeric_features)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection (inspired by ResNet but trained from scratch).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with optional projection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        
        return out


class CNNEncoder(nn.Module):
    """
    Custom CNN Encoder for 2D molecular images.
    
    Architecture:
        Input: 224×224×3 RGB image
        ├── Conv1: 3 → 64, 7×7, stride 2
        ├── BatchNorm + ReLU + MaxPool (2×2)
        ├── Layer1: 2× ResidualBlock (64 → 64)
        ├── Layer2: 2× ResidualBlock (64 → 128, stride 2)
        ├── Layer3: 2× ResidualBlock (128 → 256, stride 2)
        ├── Layer4: 2× ResidualBlock (256 → 512, stride 2)
        ├── AdaptiveAvgPool2d → (512, 1, 1)
        └── Flatten → Vector[512]
    
    Output: 512-dimensional feature vector
    """
    def __init__(self, output_dim=512):
        super(CNNEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers (custom, trained from scratch)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final feature dimension
        self.output_dim = output_dim
        
        # Initialize weights from scratch
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize all weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial block
        x = self.conv1(x)           # [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [B, 64, 56, 56]
        
        # Residual layers
        x = self.layer1(x)          # [B, 64, 56, 56]
        x = self.layer2(x)          # [B, 128, 28, 28]
        x = self.layer3(x)          # [B, 256, 14, 14]
        x = self.layer4(x)          # [B, 512, 7, 7]
        
        # Global pooling
        x = self.avgpool(x)         # [B, 512, 1, 1]
        x = torch.flatten(x, 1)     # [B, 512]
        
        return x


# ============================================================================
# MLP ENCODER - For Numeric Features
# ============================================================================
class MLPEncoder(nn.Module):
    """
    MLP Encoder for numeric molecular features.
    
    Architecture (from README_V2.md):
        Input: 23 normalized features
        ├── Linear(23 → 128)
        ├── ReLU
        ├── Dropout(0.3)
        ├── Linear(128 → 128)
        └── ReLU
    
    Output: 128-dimensional feature vector
    """
    def __init__(self, input_dim=23, hidden_dim=128, output_dim=128, dropout=0.3):
        super(MLPEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
        self.output_dim = output_dim
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.encoder(x)


# ============================================================================
# GNN ENCODER - For Molecular Graphs
# ============================================================================
class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder for molecular structure.
    
    Architecture (from README_V2.md):
        Input: Molecular graph (atoms=nodes, bonds=edges)
        ├── GCNConv(num_atom_features → 64) + ReLU
        ├── GCNConv(64 → 128) + ReLU
        ├── GCNConv(128 → 128) + ReLU
        └── global_mean_pool → Vector[128]
    
    Output: 128-dimensional graph embedding
    """
    def __init__(self, num_node_features=9, hidden_dim=64, output_dim=128, dropout=0.3):
        super(GNNEncoder, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.conv3 = GCNConv(output_dim, output_dim)
        
        self.dropout = dropout
        self.output_dim = output_dim
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        return x


# ============================================================================
# FUSION LAYER
# ============================================================================
class FusionLayer(nn.Module):
    """
    Fusion layer to combine multi-modal features.
    
    Architecture (from README_V2.md):
        Input: CNN[512] + MLP[128] + GNN[128] = 768 dim
        ├── Linear(768 → 256)
        ├── ReLU
        └── Dropout(0.3)
    
    Output: 256-dimensional fused representation
    """
    def __init__(self, cnn_dim=512, mlp_dim=128, gnn_dim=128, output_dim=256, dropout=0.3):
        super(FusionLayer, self).__init__()
        
        input_dim = cnn_dim + mlp_dim + gnn_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, cnn_features, mlp_features, gnn_features):
        combined = torch.cat([cnn_features, mlp_features, gnn_features], dim=1)
        return self.fusion(combined)


# ============================================================================
# LSTM DECODER - For SMILES Generation
# ============================================================================
class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for autoregressive SMILES generation.
    
    Architecture (from README_V2.md):
        ├── Embedding(vocab_size → 64)
        ├── LSTM(input=64, hidden=512, num_layers=2, dropout=0.3)
        └── Linear(512 → vocab_size)
    
    Parameters:
        - vocab_size: Size of SMILES vocabulary (~100 tokens)
        - embed_dim: Embedding dimension (64)
        - hidden_dim: LSTM hidden size (512)
        - num_layers: Number of LSTM layers (2)
        - context_dim: Dimension of context vector from fusion (256)
        - max_length: Maximum sequence length (150)
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=512, 
                 num_layers=2, context_dim=256, dropout=0.3, max_length=150):
        super(LSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Context projection (fused vector → initial hidden state)
        self.context_proj_h = nn.Linear(context_dim, hidden_dim * num_layers)
        self.context_proj_c = nn.Linear(context_dim, hidden_dim * num_layers)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
    
    def _init_hidden(self, context):
        """Initialize LSTM hidden state from context vector."""
        batch_size = context.size(0)
        
        # Project context to hidden states
        h = self.context_proj_h(context)
        c = self.context_proj_c(context)
        
        # Reshape to [num_layers, batch, hidden_dim]
        h = h.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c = c.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        return h, c
    
    def forward(self, context, captions=None):
        """
        Forward pass.
        
        Args:
            context: Fused feature vector [batch, context_dim]
            captions: Target tokens for teacher forcing [batch, seq_len] (training only)
        
        Returns:
            Training: Logits [batch, seq_len, vocab_size]
            Inference: Generated token indices [batch, max_length]
        """
        batch_size = context.size(0)
        hidden = self._init_hidden(context)
        
        if captions is not None:
            # Training mode with teacher forcing
            embeddings = self.embedding(captions)  # [batch, seq_len, embed_dim]
            output, _ = self.lstm(embeddings, hidden)  # [batch, seq_len, hidden_dim]
            logits = self.output_proj(output)  # [batch, seq_len, vocab_size]
            return logits
        else:
            # Inference mode (autoregressive generation)
            return self._generate(context, hidden)
    
    def _generate(self, context, hidden):
        """Autoregressive generation during inference."""
        batch_size = context.size(0)
        device = context.device
        
        # Start with <start> token (index 1)
        current_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        generated = [current_token]
        
        for _ in range(self.max_length - 1):
            embeddings = self.embedding(current_token)  # [batch, 1, embed_dim]
            output, hidden = self.lstm(embeddings, hidden)  # [batch, 1, hidden_dim]
            logits = self.output_proj(output)  # [batch, 1, vocab_size]
            
            # Greedy decoding
            current_token = logits.argmax(dim=-1)  # [batch, 1]
            generated.append(current_token)
            
            # Check if all sequences have generated <end> token (index 2)
            if (current_token == 2).all():
                break
        
        return torch.cat(generated, dim=1)


# ============================================================================
# MAIN MODEL - ChemNet-Vision
# ============================================================================
class ChemNetVisionModel(nn.Module):
    """
    ChemNet-Vision: Multimodal Neural Network for Molecule Recognition.
    
    This model combines three encoders (CNN, MLP, GNN) with a fusion layer
    and an LSTM decoder for SMILES sequence generation.
    
    Architecture:
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │  2D Image    │   │   Numeric    │   │   Graph      │
        │  (224×224)   │   │  Features    │   │   (Atoms)    │
        └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
               │                  │                  │
               ▼                  ▼                  ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ CNN Encoder  │   │ MLP Encoder  │   │ GNN Encoder  │
        │ Custom [512] │   │    [128]     │   │    [128]     │
        └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
               │                  │                  │
               └─────────────┬────┴────────────┬─────┘
                             │   FUSION        │
                             ▼                 ▼
                    ┌─────────────────────────────┐
                    │   Concatenate + Project     │
                    │        [768 → 256]          │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │     LSTM Decoder            │
                    │   [256 → 512 → vocab]       │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Output: SMILES Tokens     │
                    └─────────────────────────────┘
    
    Note: This model is trained from scratch without any pretrained weights.
    """
    def __init__(self, num_node_features=9, num_numeric_features=23, vocab_size=100):
        super(ChemNetVisionModel, self).__init__()
        
        # Dimensions
        self.cnn_dim = 512
        self.mlp_dim = 128
        self.gnn_dim = 128
        self.fusion_dim = 256
        
        # Encoders (all trained from scratch)
        self.cnn_encoder = CNNEncoder(output_dim=self.cnn_dim)
        self.mlp_encoder = MLPEncoder(
            input_dim=num_numeric_features,
            hidden_dim=128,
            output_dim=self.mlp_dim,
            dropout=0.3
        )
        self.gnn_encoder = GNNEncoder(
            num_node_features=num_node_features,
            hidden_dim=64,
            output_dim=self.gnn_dim,
            dropout=0.3
        )
        
        # Fusion layer
        self.fusion = FusionLayer(
            cnn_dim=self.cnn_dim,
            mlp_dim=self.mlp_dim,
            gnn_dim=self.gnn_dim,
            output_dim=self.fusion_dim,
            dropout=0.3
        )
        
        # LSTM Decoder
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embed_dim=64,
            hidden_dim=512,
            num_layers=2,
            context_dim=self.fusion_dim,
            dropout=0.3,
            max_length=150
        )
        
        # Property predictor (optional - for multi-task learning)
        self.property_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.vocab_size = vocab_size
    
    def forward(self, image, graph_data, numeric_features=None, captions=None):
        """
        Forward pass through the multimodal model.
        
        Args:
            image: 2D molecular image [batch, 3, 224, 224]
            graph_data: Molecular graph (PyTorch Geometric Batch)
            numeric_features: Normalized numeric features [batch, 23] (optional)
            captions: Target SMILES tokens [batch, seq_len] (training only)
        
        Returns:
            property_pred: Predicted molecular property [batch, 1]
            smiles_pred: SMILES token logits [batch, seq_len, vocab_size]
        """
        batch_size = image.size(0)
        device = image.device
        
        # 1. CNN Encoder - Image features
        cnn_features = self.cnn_encoder(image)  # [batch, 512]
        
        # 2. MLP Encoder - Numeric features
        if numeric_features is not None:
            mlp_features = self.mlp_encoder(numeric_features)  # [batch, 128]
        else:
            # Use zeros if numeric features not provided
            mlp_features = torch.zeros(batch_size, self.mlp_dim, device=device)
        
        # 3. GNN Encoder - Graph features
        gnn_features = self.gnn_encoder(graph_data)  # [batch, 128]
        
        # 4. Fusion
        fused = self.fusion(cnn_features, mlp_features, gnn_features)  # [batch, 256]
        
        # 5. Outputs
        property_pred = self.property_head(fused)  # [batch, 1]
        smiles_pred = self.decoder(fused, captions)  # [batch, seq_len, vocab_size]
        
        return property_pred, smiles_pred
    
    def generate(self, image, graph_data, numeric_features=None):
        """
        Generate SMILES from inputs (inference mode).
        
        Args:
            image: 2D molecular image [batch, 3, 224, 224]
            graph_data: Molecular graph
            numeric_features: Normalized features [batch, 23] (optional)
        
        Returns:
            Generated token indices [batch, max_length]
        """
        self.eval()
        with torch.no_grad():
            batch_size = image.size(0)
            device = image.device
            
            cnn_features = self.cnn_encoder(image)
            
            if numeric_features is not None:
                mlp_features = self.mlp_encoder(numeric_features)
            else:
                mlp_features = torch.zeros(batch_size, self.mlp_dim, device=device)
            
            gnn_features = self.gnn_encoder(graph_data)
            fused = self.fusion(cnn_features, mlp_features, gnn_features)
            
            # Generate using decoder in inference mode
            generated = self.decoder(fused, captions=None)
            
        return generated


# ============================================================================
# LEGACY SUPPORT - Backward Compatibility
# ============================================================================
class MoleculeGNN(GNNEncoder):
    """Legacy wrapper for backward compatibility."""
    def __init__(self, num_node_features, hidden_dim, output_dim, dropout=0.2):
        super().__init__(num_node_features, hidden_dim, output_dim, dropout)


class MoleculeRNN(nn.Module):
    """Legacy wrapper for backward compatibility."""
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=2):
        super(MoleculeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions=None):
        if captions is not None:
            embeddings = self.embedding(captions)
            hidden = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
            output, _ = self.gru(embeddings, hidden)
            outputs = self.fc(output)
            return outputs
        return None
