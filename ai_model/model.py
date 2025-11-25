import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torchvision import models

class MoleculeGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(MoleculeGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        return x

class MoleculeRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MoleculeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        output = self.fc(output)
        return output, hidden

class ChemNetVisionModel(nn.Module):
    def __init__(self, num_node_features, gnn_hidden, rnn_input, rnn_hidden, vocab_size):
        super(ChemNetVisionModel, self).__init__()
        
        # Image Encoder (CNN) - Using ResNet18
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, gnn_hidden) # Project to same dim as GNN
        
        # Graph Encoder (GNN)
        self.gnn = MoleculeGNN(num_node_features, gnn_hidden)
        
        # Sequence Decoder (RNN) - To generate SMILES or text info
        self.rnn = MoleculeRNN(rnn_input, rnn_hidden, vocab_size)
        
        # Fusion / Classifier
        self.classifier = nn.Linear(gnn_hidden * 2, vocab_size) # Simple fusion

    def forward(self, image, graph_data, text_input=None):
        # Image features
        img_features = self.cnn(image)
        
        # Graph features
        graph_features = self.gnn(graph_data)
        
        # Combine?
        # For now, let's just return them or fuse them
        combined = torch.cat((img_features, graph_features), dim=1)
        
        return combined

def get_model():
    # Hyperparameters
    NUM_NODE_FEATURES = 9 # Example: Atom type, etc.
    GNN_HIDDEN = 256
    RNN_INPUT = 256
    RNN_HIDDEN = 512
    VOCAB_SIZE = 100 # Example SMILES vocab size
    
    model = ChemNetVisionModel(NUM_NODE_FEATURES, GNN_HIDDEN, RNN_INPUT, RNN_HIDDEN, VOCAB_SIZE)
    return model
