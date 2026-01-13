"""
Fast Training Script for Molecule NLP Model
===========================================
Trains a simple LSTM-based sentence generator to talk about molecules.

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

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.model import get_nlp_model

class MoleculeTextDataset(Dataset):
    def __init__(self, csv_data_path, chunks_data_path, vocab_path=None, max_samples=2000):
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
        
        # Add pairs from structured data (Short templates: 10-12 words)
        for d in self.structured_data[:1000]:
            name = str(d.get('Name', 'Unknown'))
            weight = f"{float(d.get('Molecular Weight', 0)):.2f}" if d.get('Molecular Weight') else "N/A"
            self.pairs.append((f"What is {name}?", f"This is {name}, a molecule weighing {weight} units."))
            self.pairs.append((f"Info on {name}", f"{name} weights {weight} and is found in our database."))

        # Add pairs from chunks data (Richer language)
        for chunk in self.text_chunks[:500]:
            content = chunk.get('content', '')
            title = chunk.get('metadata', {}).get('title', '').replace(' - Wikipedia', '')
            
            if title and content:
                # Extract first sentence for the 10-12 word limit
                first_sentence = content.split('.')[0].strip()
                if len(first_sentence.split()) > 15:
                    first_sentence = " ".join(first_sentence.split()[:12]) + "."
                
                if title.upper() not in ["UNKNOWN", ""]:
                    self.pairs.append((f"Tell me about {title}", first_sentence))

        self.pairs = self.pairs[:max_samples]
        
        # Simple vocab: characters
        all_text = "".join([f"{p[0]} {p[1]}" for p in self.pairs])
        self.chars = ['<pad>', '<start>', '<end>'] + sorted(list(set(all_text)))
        self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
        self.idx_to_char = {i: char for i, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        if vocab_path:
            with open(vocab_path, 'w') as f:
                json.dump(self.char_to_idx, f)

    def __len__(self):
        return len(self.pairs)

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def __getitem__(self, idx):
        q, r = self.pairs[idx]
        input_ids = self.encode(q)
        target_ids = self.encode(r)
        return torch.tensor(input_ids), torch.tensor(target_ids)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    csv_path = os.path.join(base_dir, 'data', 'processed', 'molecules_processed.csv')
    chunks_path = os.path.join(base_dir, 'data', 'chunks.json')
    vocab_path = os.path.join(base_dir, 'models', 'nlp_vocab.json')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    dataset = MoleculeTextDataset(csv_path, chunks_path, vocab_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    model = get_nlp_model(dataset.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"Starting fast training on {device}...")
    for epoch in range(20): # increased to 20 for richer data
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            # For simplicity, we just use the model to predict next tokens in the response
            # In a real Seq2Seq we'd use an encoder-decoder, but this is a "simple" LSTM
            logits, _ = model(targets[:, :-1])
            loss = criterion(logits.reshape(-1, dataset.vocab_size), targets[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    model_path = os.path.join(base_dir, 'models', 'nlp_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
