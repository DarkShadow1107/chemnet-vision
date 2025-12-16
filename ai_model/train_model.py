"""
ChemNet-Vision Model Training Script
=====================================
This script trains a custom neural network model (from scratch, no pretraining) 
for molecule recognition using:
- CNN (Custom ResNet-style) for 2D image feature extraction
- MLP for numeric molecular features
- GNN (Graph Convolutional Network) for molecular graph processing
- LSTM for SMILES sequence generation

Uses preprocessed data from Etapa 3.

Author: Alexandru Gabriel
Institution: POLITEHNICA București – FIIR
Discipline: Rețele Neuronale
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
import argparse
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from ai_model.model import get_model

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# List of numeric feature columns (23 features)
NUMERIC_FEATURES = [
    'Molecular Weight', 'Targets', 'Bioactivities', 'AlogP', 
    'Polar Surface Area', 'HBA', 'HBD', '#RO5 Violations', 
    '#Rotatable Bonds', 'QED Weighted', 'Aromatic Rings', 
    'Heavy Atoms', 'Np Likeness Score',
    'MolWeight_RDKit', 'LogP_RDKit', 'TPSA_RDKit', 
    'NumHDonors_RDKit', 'NumHAcceptors_RDKit', 'NumRotatableBonds_RDKit',
    'NumAromaticRings_RDKit', 'FractionCSP3', 'NumHeteroatoms', 'RingCount'
]


def smiles_to_graph(smiles):
    """Convert SMILES string to PyTorch Geometric graph."""
    if not smiles or pd.isna(smiles):
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features: atom properties (9 features)
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
    """Dataset for molecule images, graphs, numeric features, and SMILES."""
    
    def __init__(self, csv_file, img_dir, transform=None, vocab=None, max_seq_len=150):
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
        
        # Find available numeric features in dataframe
        self.available_features = [f for f in NUMERIC_FEATURES if f in self.df.columns]
        
        # Also check for normalized versions
        normalized_features = [f"{f}_normalized" for f in NUMERIC_FEATURES]
        self.normalized_features = [f for f in normalized_features if f in self.df.columns]
        
        # Use normalized if available, otherwise use original
        if len(self.normalized_features) >= 20:
            self.feature_cols = self.normalized_features
        else:
            self.feature_cols = self.available_features
        
        # Count images
        if 'has_image' in self.df.columns:
            num_with_images = self.df['has_image'].sum()
        else:
            num_with_images = 0
        
        print(f"Dataset loaded: {len(self.df)} molecules")
        print(f"  - With images: {num_with_images}")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Numeric features: {len(self.feature_cols)}")
        
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
        
        # Extract numeric features
        numeric_features = []
        for col in self.feature_cols:
            val = row.get(col, 0)
            if pd.isna(val):
                val = 0.0
            numeric_features.append(float(val))
        
        # Pad to 23 features if needed
        while len(numeric_features) < 23:
            numeric_features.append(0.0)
        numeric_features = numeric_features[:23]  # Truncate if more
        
        numeric_tensor = torch.tensor(numeric_features, dtype=torch.float)
        
        # Tokenize SMILES
        tokens = [self.vocab['<start>']]
        tokens += [self.vocab.get(c, 0) for c in str(smiles)]
        tokens.append(self.vocab['<end>'])
        
        # Pad or truncate
        if len(tokens) < self.max_seq_len:
            tokens += [self.vocab['<pad>']] * (self.max_seq_len - len(tokens))
        else:
            tokens = tokens[:self.max_seq_len]
        
        return image, graph, numeric_tensor, torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images, graphs, numeric_features, captions = zip(*batch)
    images = torch.stack(images)
    graphs = Batch.from_data_list(graphs)
    numeric_features = torch.stack(numeric_features)
    captions = torch.stack(captions)
    return images, graphs, numeric_features, captions


class Trainer:
    """Model trainer with logging and checkpointing."""
    
    def __init__(self, model, train_loader, val_loader, device, vocab_size,
                lr=0.001, checkpoint_dir='saved_models', models_dir=None):
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

        self.models_dir = models_dir
        if self.models_dir is not None:
            os.makedirs(self.models_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """Train for one epoch.

        Returns:
            (avg_loss, token_accuracy)
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        correct_tokens = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for images, graphs, numeric_features, captions in pbar:
            images = images.to(self.device)
            graphs = graphs.to(self.device)
            numeric_features = numeric_features.to(self.device)
            captions = captions.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Teacher forcing (next-token prediction):
                # input tokens:  [<start>, t1, t2, ...]
                # target tokens: [t1, t2, ..., <end>]
                captions_in = captions[:, :-1]
                targets = captions[:, 1:]

                _, smiles_pred = self.model(images, graphs, numeric_features, captions_in)

                # Calculate loss vs next-token targets (ignore <pad>=0)
                loss = self.criterion(
                    smiles_pred.reshape(-1, self.vocab_size),
                    targets.reshape(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Token-level accuracy (exclude <pad> = 0)
                with torch.no_grad():
                    pred = torch.argmax(smiles_pred, dim=-1)
                    mask = targets != 0
                    correct_tokens += int(((pred == targets) & mask).sum().item())
                    total_tokens += int(mask.sum().item())
                
                total_loss += loss.item()
                num_batches += 1
                
                acc = (correct_tokens / max(total_tokens, 1))
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
            except Exception as e:
                print(f"\nWarning: Batch error - {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        token_acc = correct_tokens / max(total_tokens, 1)
        return avg_loss, float(token_acc)
    
    def validate(self, epoch):
        """Validate the model.

        Returns:
            (avg_loss, token_accuracy)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        correct_tokens = 0
        total_tokens = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, graphs, numeric_features, captions in pbar:
                images = images.to(self.device)
                graphs = graphs.to(self.device)
                numeric_features = numeric_features.to(self.device)
                captions = captions.to(self.device)
                
                try:
                    captions_in = captions[:, :-1]
                    targets = captions[:, 1:]

                    _, smiles_pred = self.model(images, graphs, numeric_features, captions_in)

                    loss = self.criterion(
                        smiles_pred.reshape(-1, self.vocab_size),
                        targets.reshape(-1)
                    )

                    pred = torch.argmax(smiles_pred, dim=-1)
                    mask = targets != 0
                    correct_tokens += int(((pred == targets) & mask).sum().item())
                    total_tokens += int(mask.sum().item())
                    
                    total_loss += loss.item()
                    num_batches += 1

                    acc = (correct_tokens / max(total_tokens, 1))
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
                except Exception as e:
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        self.val_losses.append(avg_loss)
        token_acc = correct_tokens / max(total_tokens, 1)
        return avg_loss, float(token_acc)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'vocab_size': self.vocab_size,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

            # Keep Etapa 5 deliverable in sync: models/trained_model.pth should be the best checkpoint
            if self.models_dir is not None:
                try:
                    trained_out = os.path.join(self.models_dir, 'trained_model.pth')
                    torch.save(checkpoint, trained_out)
                except Exception:
                    pass
    
    def train(self, num_epochs, *, start_epoch=1, early_stopping=False, early_stopping_patience=5, history_csv_path=None):
        """Full training loop.

        Args:
            num_epochs: number of epochs to run
            early_stopping: stop if val_loss doesn't improve for `early_stopping_patience` epochs
            early_stopping_patience: patience for early stopping
            history_csv_path: optional CSV path to append epoch history
        """
        print(f"\n{'='*60}")
        print(f"Starting Training - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Optional CSV header
        if history_csv_path is not None:
            os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
            expected_fields = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr', 'timestamp']
            if os.path.exists(history_csv_path):
                try:
                    with open(history_csv_path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    if first_line and first_line != ','.join(expected_fields):
                        base, ext = os.path.splitext(history_csv_path)
                        history_csv_path = f"{base}_v2{ext or '.csv'}"
                        print(f"[i] Existing training history has a different header; writing to {history_csv_path}")
                except Exception:
                    pass

            if not os.path.exists(history_csv_path):
                with open(history_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=expected_fields)
                    writer.writeheader()

        bad_epochs = 0
        best_epoch = 0
        
        if start_epoch > num_epochs:
            print(f"Start epoch {start_epoch} is > num_epochs {num_epochs}; nothing to do.")
            return

        for epoch in range(start_epoch, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)

            self.train_accuracies.append(float(train_acc))
            self.val_accuracies.append(float(val_acc))
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            if is_best:
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1

            # Append to history CSV
            if history_csv_path is not None:
                with open(history_csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr', 'timestamp'],
                    )
                    writer.writerow({
                        'epoch': epoch,
                        'train_loss': f'{train_loss:.6f}',
                        'train_accuracy': f'{train_acc:.6f}',
                        'val_loss': f'{val_loss:.6f}',
                        'val_accuracy': f'{val_acc:.6f}',
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.8f}",
                        'timestamp': datetime.now().isoformat(timespec='seconds'),
                    })

            # Print summary (Etapa 5 README-friendly)
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
            )
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")

            if early_stopping and bad_epochs >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best epoch: {best_epoch}, best val_loss: {self.best_val_loss:.4f})"
                )
                break
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("ChemNet-Vision Model Training")
    print("Custom Model (No Pretraining)")
    print("="*60)
    
    parser = argparse.ArgumentParser(description='ChemNet-Vision training (Etapa 5)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_seq_len', type=int, default=150)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--augment', action='store_true', help='Enable mild, domain-relevant augmentations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--resume',
        default=None,
        help=(
            'Resume training from a checkpoint (.pth). Example: saved_models/checkpoint_latest.pth. '
            'If provided, loads model+optimizer(+scheduler when available) and continues from epoch+1.'
        ),
    )
    args = parser.parse_args()

    if args.epochs < 10:
        raise SystemExit('Etapa 5 requires minimum 10 epochs. Use --epochs >= 10')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    DOCS_DIR = os.path.join(BASE_DIR, 'docs')
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Image transforms
    # Note: augmentations are mild and chemistry-safe (no heavy rotation-only policy).
    if args.augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomPerspective(distortion_scale=0.08, p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
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
        max_seq_len=args.max_seq_len
    )
    
    val_dataset = MoleculeDataset(
        csv_file=os.path.join(DATA_DIR, 'validation', 'validation.csv'),
        img_dir=os.path.join(DATA_DIR, '2d_images'),
        transform=transform,
        vocab=train_dataset.vocab,  # Use same vocab
        max_seq_len=args.max_seq_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model (custom, no pretraining)
    vocab_size = train_dataset.vocab_size + 10  # Add buffer
    num_numeric_features = 23  # From preprocessing
    print(f"\nCreating custom model (no pretraining)...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Numeric features: {num_numeric_features}")
    model = get_model(
        vocab_size=vocab_size,
        num_node_features=9,
        num_numeric_features=num_numeric_features
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save vocabulary (both saved_models/ and models/ for Etapa 5)
    vocab_path = os.path.join(SAVED_MODELS_DIR, 'vocab.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    print(f"Vocabulary saved to: {vocab_path}")

    vocab_models_path = os.path.join(MODELS_DIR, 'vocab.json')
    if not os.path.exists(vocab_models_path):
        with open(vocab_models_path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset.vocab, f, indent=2)
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        vocab_size=vocab_size,
        lr=args.lr,
        checkpoint_dir=SAVED_MODELS_DIR,
        models_dir=MODELS_DIR,
    )

    # Optional resume
    start_epoch = 1
    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(BASE_DIR, resume_path)

        if os.path.exists(resume_path):
            print(f"\nResuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)

            # Support both full checkpoint dict and raw state_dict
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                trainer.model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    try:
                        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    except Exception:
                        pass

                trainer.train_losses = list(ckpt.get('train_losses', []))
                trainer.val_losses = list(ckpt.get('val_losses', []))
                trainer.train_accuracies = list(ckpt.get('train_accuracies', []))
                trainer.val_accuracies = list(ckpt.get('val_accuracies', []))
                trainer.best_val_loss = float(ckpt.get('best_val_loss', trainer.best_val_loss))

                last_epoch = int(ckpt.get('epoch', 0))
                start_epoch = last_epoch + 1
                print(f"Resuming at epoch {start_epoch} (last completed: {last_epoch})")
            else:
                # raw state_dict
                trainer.model.load_state_dict(ckpt)
                print("Loaded raw model state_dict; optimizer/scheduler not restored.")
        else:
            print(f"[!] Resume checkpoint not found: {resume_path}. Starting from scratch.")

    # Save an explicit untrained checkpoint (Etapa 4/5 checklist)
    untrained_path = os.path.join(MODELS_DIR, 'untrained_model.pth')
    if not os.path.exists(untrained_path):
        torch.save({'model_state_dict': model.state_dict(), 'vocab_size': vocab_size}, untrained_path)

    # Save hyperparameters
    hyperparams = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_seq_len': args.max_seq_len,
        'num_workers': args.num_workers,
        'early_stopping': bool(args.early_stopping),
        'early_stopping_patience': args.patience,
        'scheduler': 'ReduceLROnPlateau',
        'augment': bool(args.augment),
        'seed': args.seed,
    }
    try:
        import yaml  # type: ignore

        with open(os.path.join(RESULTS_DIR, 'hyperparameters.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(hyperparams, f, sort_keys=False)
    except Exception:
        with open(os.path.join(RESULTS_DIR, 'hyperparameters.json'), 'w', encoding='utf-8') as f:
            json.dump(hyperparams, f, indent=2)

    history_csv_path = os.path.join(RESULTS_DIR, 'training_history.csv')

    def _save_training_plots() -> None:
        # Loss curve
        try:
            if len(trainer.train_losses) == 0:
                return

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(trainer.train_losses) + 1), trainer.train_losses, label='train_loss')
            plt.plot(range(1, len(trainer.val_losses) + 1), trainer.val_losses, label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(DOCS_DIR, 'loss_curve.png'))
            plt.close()
        except Exception:
            pass

        # Learning curves (loss + accuracy)
        try:
            epochs_range = list(range(1, len(trainer.train_losses) + 1))

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, trainer.train_losses, label='train_loss')
            plt.plot(epochs_range, trainer.val_losses, label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Learning Curve (Loss)')
            plt.legend()

            plt.subplot(1, 2, 2)
            if len(trainer.train_accuracies) == len(epochs_range):
                plt.plot(epochs_range, trainer.train_accuracies, label='train_accuracy')
            if len(trainer.val_accuracies) == len(epochs_range):
                plt.plot(epochs_range, trainer.val_accuracies, label='val_accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Token Accuracy')
            plt.title('Learning Curve (Accuracy)')
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(DOCS_DIR, 'learning_curves.png'))
            plt.close()
        except Exception:
            pass

    try:
        trainer.train(
            num_epochs=args.epochs,
            start_epoch=start_epoch,
            early_stopping=args.early_stopping,
            early_stopping_patience=args.patience,
            history_csv_path=history_csv_path,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (KeyboardInterrupt). Saving artifacts from completed epochs...")
    finally:
        # Keep Etapa 5 deliverable copy in models/ even if training is interrupted
        best_ckpt = os.path.join(SAVED_MODELS_DIR, 'checkpoint_best.pth')
        trained_out = os.path.join(MODELS_DIR, 'trained_model.pth')
        if os.path.exists(best_ckpt):
            try:
                ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
                torch.save(ckpt, trained_out)
            except Exception:
                pass

        _save_training_plots()

    print("\nEtapa 5 artifacts:")
    print(f"- models/untrained_model.pth")
    print(f"- models/trained_model.pth")
    print(f"- results/training_history.csv")
    print(f"- results/hyperparameters.yaml (or .json)")
    print(f"- docs/loss_curve.png")
    print(f"- docs/learning_curves.png")


if __name__ == "__main__":
    main()
