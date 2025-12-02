"""
ChemNet-Vision Model Training Script
=====================================
This script trains a neural network model for molecule recognition using:
- CNN (ResNet18) for 2D image feature extraction
- GNN (Graph Attention Network) for molecular graph processing
- RNN (GRU) for SMILES sequence generation

Uses preprocessed data from Etapa 3.
"""

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
import sys
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from ai_model.model import get_model

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric graph."""
    if not smiles or pd.isna(smiles):
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features: atom properties
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass(),
            atom.GetExplicitValence(),
            atom.GetImplicitValence(),
            int(atom.IsInRing())
        ])
    
    if len(atom_features) == 0:
        return None
        
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices: bonds
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    
    if len(edge_indices) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


class MoleculeDataset(Dataset):
    """Dataset for molecule images and SMILES."""
    
    def __init__(self, csv_file, img_dir, transform=None, vocab=None, max_seq_len=100):
        """
        Args:
            csv_file: Path to CSV with molecule data (includes image_path column)
            img_dir: Directory containing 2D images (fallback)
            transform: Image transforms
            vocab: Character vocabulary for SMILES
            max_seq_len: Maximum SMILES sequence length
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # Build or use provided vocabulary
        self.vocab = vocab or self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # Count images
        if 'has_image' in self.df.columns:
            num_with_images = self.df['has_image'].sum()
        else:
            num_with_images = 0
        
        print(f"Dataset loaded: {len(self.df)} molecules, {num_with_images} with images, vocab size: {self.vocab_size}")
        
    def _build_vocab(self):
        """Build character vocabulary from SMILES strings."""
        chars = set()
        smiles_col = 'Smiles' if 'Smiles' in self.df.columns else 'SMILES'
        
        for smiles in self.df[smiles_col].dropna():
            chars.update(list(str(smiles)))
        
        vocab = {char: i + 3 for i, char in enumerate(sorted(chars))}
        vocab['<pad>'] = 0
        vocab['<start>'] = 1
        vocab['<end>'] = 2
        return vocab
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['Name']
        
        # Get SMILES (handle different column names)
        smiles = row.get('Smiles') if 'Smiles' in row else row.get('SMILES', '')
        if pd.isna(smiles):
            smiles = ''
        
        # Load image from image_path column or construct path from name
        img_path = row.get('image_path', '')
        if pd.isna(img_path) or not img_path:
            img_path = os.path.join(self.img_dir, f"{name}.png")
        
        if img_path and os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        # Create molecular graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            graph = Data(x=torch.zeros(1, 9), edge_index=torch.zeros(2, 0, dtype=torch.long))
        
        # Tokenize SMILES
        tokens = [self.vocab['<start>']]
        tokens += [self.vocab.get(c, 0) for c in str(smiles)]
        tokens.append(self.vocab['<end>'])
        
        # Pad or truncate
        if len(tokens) < self.max_seq_len:
            tokens += [self.vocab['<pad>']] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        
        return image, graph, torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images, graphs, captions = zip(*batch)
    images = torch.stack(images)
    graphs = Batch.from_data_list(graphs)
    captions = torch.stack(captions)
    return images, graphs, captions


class Trainer:
    """Model trainer with logging and checkpointing."""
    
    def __init__(self, model, train_loader, val_loader, device, vocab_size, 
                 lr=0.001, checkpoint_dir='saved_models'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.vocab_size = vocab_size
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for images, graphs, captions in pbar:
            images = images.to(self.device)
            graphs = graphs.to(self.device)
            captions = captions.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                _, smiles_pred = self.model(images, graphs, captions)
                
                # Calculate loss
                loss = self.criterion(
                    smiles_pred.view(-1, self.vocab_size),
                    captions.view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            except Exception as e:
                print(f"\nWarning: Batch error - {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, graphs, captions in pbar:
                images = images.to(self.device)
                graphs = graphs.to(self.device)
                captions = captions.to(self.device)
                
                try:
                    _, smiles_pred = self.model(images, graphs, captions)
                    
                    loss = self.criterion(
                        smiles_pred.view(-1, self.vocab_size),
                        captions.view(-1)
                    )
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                except Exception as e:
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_size': self.vocab_size,
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
    
    def train(self, num_epochs):
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting Training - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            print()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("ChemNet-Vision Model Training")
    print("="*60)
    
    # Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    CONFIG = {
        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'max_seq_len': 150,
        'num_workers': 0,  # Set to 0 for Windows compatibility
    }
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    
    train_dataset = MoleculeDataset(
        csv_file=os.path.join(DATA_DIR, 'train', 'train.csv'),
        img_dir=os.path.join(DATA_DIR, '2d_images'),
        transform=transform,
        max_seq_len=CONFIG['max_seq_len']
    )
    
    val_dataset = MoleculeDataset(
        csv_file=os.path.join(DATA_DIR, 'validation', 'validation.csv'),
        img_dir=os.path.join(DATA_DIR, '2d_images'),
        transform=transform,
        vocab=train_dataset.vocab,  # Use same vocab
        max_seq_len=CONFIG['max_seq_len']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    vocab_size = train_dataset.vocab_size + 10  # Add buffer
    print(f"\nCreating model with vocab_size={vocab_size}")
    model = get_model(vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save vocabulary
    vocab_path = os.path.join(BASE_DIR, 'saved_models', 'vocab.json')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        vocab_size=vocab_size,
        lr=CONFIG['learning_rate'],
        checkpoint_dir=os.path.join(BASE_DIR, 'saved_models')
    )
    
    trainer.train(num_epochs=CONFIG['num_epochs'])
    
    # Save final model
    final_path = os.path.join(BASE_DIR, 'saved_models', 'chemnet_vision_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
