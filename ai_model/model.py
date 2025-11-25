import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torchvision import models

class MoleculeGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, dropout=0.2):
        super(MoleculeGNN, self).__init__()
        # Using GAT (Graph Attention Network) for better feature extraction
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        return x

class MoleculeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=2):
        super(MoleculeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions=None):
        # features: [batch_size, hidden_size] (from CNN or GNN)
        # captions: [batch_size, max_seq_len] (target SMILES indices)
        
        if captions is not None:
            # Training mode with Teacher Forcing
            embeddings = self.embedding(captions) # [batch, seq_len, hidden]
            
            # Prepare initial hidden state from features
            # We repeat features to match num_layers
            hidden = features.unsqueeze(0).repeat(self.num_layers, 1, 1) # [layers, batch, hidden]
            
            output, _ = self.gru(embeddings, hidden)
            outputs = self.fc(output)
            return outputs
        else:
            # Inference mode (Greedy search)
            # This part would need a loop to generate token by token
            pass
        return None

class ChemNetVisionModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, vocab_size):
        super(ChemNetVisionModel, self).__init__()
        
        # Image Encoder (CNN) - ResNet18
        # We remove the last FC layer to get features
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.cnn_fc = nn.Linear(resnet.fc.in_features, hidden_dim)
        
        # Graph Encoder (GNN)
        self.gnn = MoleculeGNN(num_node_features, hidden_dim, hidden_dim)
        
        # Sequence Decoder (RNN)
        self.rnn = MoleculeRNN(hidden_dim, hidden_dim, vocab_size)
        
        # Property Predictor (Regression/Classification)
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Example: Predicting 1 property
        )

    def forward(self, image, graph_data, captions=None):
        # 1. Image Branch
        img_features = self.cnn(image) # [batch, 512, 1, 1]
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.cnn_fc(img_features) # [batch, hidden_dim]
        
        # 2. Graph Branch
        graph_features = self.gnn(graph_data) # [batch, hidden_dim]
        
        # 3. Fusion
        combined_features = torch.cat((img_features, graph_features), dim=1)
        
        # 4. Outputs
        # Property Prediction
        property_pred = self.property_head(combined_features)
        
        # SMILES Generation (using Image features primarily, or fused)
        # Usually for "Recognition", we use Image -> SMILES
        smiles_pred = self.rnn(img_features, captions)
        
        return property_pred, smiles_pred

def get_model(vocab_size=100):
    # Hyperparameters
    NUM_NODE_FEATURES = 9 # Atom features (atomic num, degree, etc.)
    HIDDEN_DIM = 256
    
    model = ChemNetVisionModel(NUM_NODE_FEATURES, HIDDEN_DIM, vocab_size)
    return model
