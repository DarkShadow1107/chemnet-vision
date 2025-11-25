import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import os
import numpy as np
from rdkit import Chem
from model import get_model

# --- Helper Functions for Graph Conversion ---
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features: Atomic number (one-hot or integer)
    # Simplified: just atomic number
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization(),
            int(atom.GetIsAromatic()),
            atom.GetMass(),
            atom.GetExplicitValence(),
            atom.GetImplicitValence(),
            int(atom.IsInRing())
        ])
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge features: Bond type
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        # Simplified edge attr
        edge_attrs.append(bond.GetBondTypeAsDouble())
        edge_attrs.append(bond.GetBondTypeAsDouble())
        
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

# --- Dataset Class ---
class MoleculeDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, vocab=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.vocab = vocab or self.build_vocab()
        
    def build_vocab(self):
        chars = set()
        for item in self.data:
            smiles = item.get('SMILES', '')
            chars.update(list(smiles))
        vocab = {char: i+1 for i, char in enumerate(sorted(chars))} # 0 is padding
        vocab['<pad>'] = 0
        vocab['<start>'] = len(vocab)
        vocab['<end>'] = len(vocab) + 1
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        name = item['Name']
        smiles = item.get('SMILES', '')
        
        # 1. Load Image
        img_path = os.path.join(self.img_dir, f"{name}.png")
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Placeholder black image
            image = Image.new('RGB', (224, 224), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        # 2. Create Graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            # Dummy graph
            graph = Data(x=torch.zeros(1, 9), edge_index=torch.zeros(2, 0, dtype=torch.long))

        # 3. Tokenize SMILES
        tokens = [self.vocab['<start>']] + [self.vocab.get(c, 0) for c in smiles] + [self.vocab['<end>']]
        # Pad to fixed length (e.g., 100)
        max_len = 100
        if len(tokens) < max_len:
            tokens += [self.vocab['<pad>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
            
        return image, graph, torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    images, graphs, captions = zip(*batch)
    images = torch.stack(images)
    graphs = Batch.from_data_list(graphs)
    captions = torch.stack(captions)
    return images, graphs, captions

def train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MoleculeDataset(
        json_file=os.path.join('data', 'molecules.json'),
        img_dir=os.path.join('data', '2d_images'),
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    vocab_size = len(dataset.vocab) + 2 # +2 for safety
    model = get_model(vocab_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_smiles = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding
    
    print("Starting training...")
    model.train()
    
    for epoch in range(5): # 5 Epochs for demo
        total_loss = 0
        for images, graphs, captions in dataloader:
            images = images.to(device)
            graphs = graphs.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # We don't have property labels in this dummy dataset, so we ignore property_pred
            _, smiles_pred = model(images, graphs, captions)
            
            # Calculate loss
            # smiles_pred: [batch, seq_len, vocab_size]
            # captions: [batch, seq_len]
            # We need to flatten for CrossEntropy
            loss = criterion_smiles(smiles_pred.view(-1, vocab_size), captions.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'saved_models/chemnet_vision.pth')
    print("Model saved to saved_models/chemnet_vision.pth")

if __name__ == "__main__":
    train()
