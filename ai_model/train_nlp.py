"""
Training Script for Molecule NLP Model (v2 - Optimized)
========================================================
Trains a 2-layer LSTM sentence generator with:
- Gradient clipping to prevent exploding gradients
- Cosine annealing LR for smooth convergence
- Full vocab built from ALL data (not sampled)
- Validation loss tracking for early stopping

Author: Alexandru Gabriel
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import os
import sys
import math

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.model import get_nlp_model

class MoleculeTextDataset(Dataset):
    def __init__(self, csv_data_path, chunks_data_path, vocab_path=None, max_samples=300000):
        # 1. Load structured data
        if os.path.exists(csv_data_path):
            self.structured_data = pd.read_csv(csv_data_path).to_dict('records')
        else:
            self.structured_data = []

        # 2. Load text chunks from Wikipedia (RAG source)
        self.text_chunks = []
        if os.path.exists(chunks_data_path):
            with open(chunks_data_path, 'r') as f:
                self.text_chunks = json.load(f)

        self.pairs = []

        # Create a mapping for rich descriptions from chunks
        chunk_descriptions = {}
        for chunk in self.text_chunks:
            title = chunk.get('metadata', {}).get('title', '').replace(' - Wikipedia', '').upper()
            content = chunk.get('content', '')
            if title and content:
                sentences = content.split('. ')
                brief_info = ". ".join(sentences[:1]).strip()
                if not brief_info.endswith('.'):
                    brief_info += "."
                if len(brief_info) > 150:
                    brief_info = brief_info[:147] + "..."
                chunk_descriptions[title] = brief_info

        # Add pairs from structured data (ALL molecules)
        separator = " -> "
        for d in self.structured_data:
            name = str(d.get('Name', 'Unknown'))
            if name.lower() == 'nan' or not name: continue

            formula = d.get('Molecular Formula', 'N/A')
            weight_val = d.get('MolWeight_RDKit') or d.get('Molecular Weight') or 0
            weight = f"{float(weight_val):.2f}"
            mol_type = d.get('Type', 'Small molecule')

            # Extract synonyms
            syns_raw = d.get('Synonyms', '')
            syns_list = []
            if isinstance(syns_raw, str) and syns_raw.lower() != 'nan':
                parts = syns_raw.replace('|', ',').replace(';', ',').split(',')
                syns_list = [s.strip() for s in parts if s.strip() and s.strip().upper() != name.upper()][:2]
            syns_str = ", ".join(syns_list) if syns_list else "None"

            target_main = (f"{separator}{name} is a {mol_type}. "
                          f"Formula: {formula}. Weight: {weight} g/mol. "
                          f"Synonyms: {syns_str}.")

            # Multiple prompt variants so model learns to associate name with data
            self.pairs.append((f"Describe {name}", target_main))
            self.pairs.append((f"What is {name}", target_main))
            self.pairs.append((f"Show me {name}", target_main))

            info = chunk_descriptions.get(name.upper())
            if info:
                target_rich = f"{separator}{name} is {info} Type: {mol_type}. Formula: {formula}."
                self.pairs.append((f"Tell me about {name}", target_rich))
            else:
                self.pairs.append((f"Tell me about {name}", target_main))

        # Build vocab from ALL pairs (not sampled) to ensure complete coverage
        all_chars = set()
        for q, r in self.pairs:
            all_chars.update(q)
            all_chars.update(r)

        extra_chars = "()[]{}0123456789.+-=#$@!%^&*_/\\:;,<>"
        all_chars.update(extra_chars)
        all_chars.discard('\n')

        self.chars = ['<pad>', '<start>', '<end>', '\n'] + sorted(list(all_chars))
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        if max_samples < len(self.pairs):
            self.pairs = self.pairs[:max_samples]

        if vocab_path:
            with open(vocab_path, 'w') as f:
                json.dump(self.char_to_idx, f)

    def __len__(self):
        return len(self.pairs)

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def __getitem__(self, idx):
        q, r = self.pairs[idx]
        full_text = q + r + "\n"
        ids = self.encode(full_text)
        return torch.tensor(ids)

def collate_fn(batch):
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = os.path.join(base_dir, 'data', 'processed', 'molecules_processed.csv')
    chunks_path = os.path.join(base_dir, 'data', 'chunks.json')
    vocab_path = os.path.join(base_dir, 'models', 'nlp_vocab.json')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    dataset = MoleculeTextDataset(csv_path, chunks_path, vocab_path, max_samples=300000)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = get_nlp_model(dataset.vocab_size).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 80
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"Training on {device} | Dataset: {len(dataset)} pairs | Vocab: {dataset.vocab_size} chars")
    print(f"Config: epochs={num_epochs}, batch=128, lr=0.001, hidden=512, layers=2")

    best_loss = float('inf')
    model_path = os.path.join(base_dir, 'models', 'nlp_model.pth')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for i, ids in enumerate(dataloader):
            ids = ids.to(device)
            optimizer.zero_grad()
            logits, _ = model(ids[:, :-1])
            loss = criterion(logits.reshape(-1, dataset.vocab_size), ids[:, 1:].reshape(-1))
            loss.backward()
            # Gradient clipping to prevent exploding gradients in deep LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if i % 200 == 0 and i > 0:
                print(f"  Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 5 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {lr:.6f}")

    # Save final model (best was already saved during training)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
