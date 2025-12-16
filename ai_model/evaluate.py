"""Etapa 5 - Evaluation for ChemNet-Vision (PyTorch)

This project is an autoregressive SMILES generator (sequence model). For Etapa 5
we compute *token-level* metrics on the teacher-forced outputs and generate
required deliverables:

- results/test_metrics.json
- docs/confusion_matrix.png

Notes:
- Metrics exclude <pad> tokens.
- Confusion matrix is shown for top-K most frequent token classes (+ OTHER).
- Use --max_batches for a fast debug run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms

# Ensure the repository root is on sys.path so `ai_model.*` imports work when
# running `python ai_model/evaluate.py` from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Local imports from this repo
from ai_model.model import get_model
from ai_model.train_model import MoleculeDataset, collate_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='ChemNet-Vision evaluation (Etapa 5)')
    p.add_argument('--model', default=os.path.join('models', 'trained_model.pth'))
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--max_seq_len', type=int, default=150)
    p.add_argument('--top_tokens', type=int, default=25)
    p.add_argument('--max_batches', type=int, default=None)
    return p.parse_args()


def _decode_tokens(tokens: list[int], idx_to_char: dict[int, str], start_id: int, end_id: int, pad_id: int) -> str:
    out: list[str] = []
    for t in tokens:
        if t == end_id:
            break
        if t in (start_id, pad_id):
            continue
        out.append(idx_to_char.get(int(t), ''))
    return ''.join(out)


def main() -> None:
    args = parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    saved_models_dir = os.path.join(base_dir, 'saved_models')
    results_dir = os.path.join(base_dir, 'results')
    docs_dir = os.path.join(base_dir, 'docs')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    # Resolve vocab
    vocab_path = None
    for candidate in [
        os.path.join(models_dir, 'vocab.json'),
        os.path.join(saved_models_dir, 'vocab.json'),
    ]:
        if os.path.exists(candidate):
            vocab_path = candidate
            break

    if vocab_path is None:
        raise SystemExit('Missing vocab.json. Run training first (ai_model/train_model.py) to generate it.')

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    idx_to_char = {int(v): k for k, v in vocab.items()}
    pad_id = int(vocab.get('<pad>', 0))
    start_id = int(vocab.get('<start>', 1))
    end_id = int(vocab.get('<end>', 2))

    # Resolve checkpoint path
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)

    if not os.path.exists(model_path):
        # fallback to saved_models
        fallback = os.path.join(saved_models_dir, 'checkpoint_best.pth')
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise SystemExit(f'Model checkpoint not found: {args.model}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    vocab_size = int(checkpoint.get('vocab_size', len(vocab) + 10))

    model = get_model(vocab_size=vocab_size, num_node_features=9, num_numeric_features=23).to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MoleculeDataset(
        csv_file=os.path.join(data_dir, 'test', 'test.csv'),
        img_dir=os.path.join(data_dir, '2d_images'),
        transform=transform,
        vocab=vocab,
        max_seq_len=args.max_seq_len,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    all_true: list[int] = []
    all_pred: list[int] = []

    # optional validity
    try:
        from rdkit import Chem  # type: ignore

        rdkit_available = True
    except Exception:
        rdkit_available = False

    valid_count = 0
    total_smiles = 0

    with torch.no_grad():
        for batch_idx, (images, graphs, numeric_features, captions) in enumerate(test_loader, start=1):
            images = images.to(device)
            graphs = graphs.to(device)
            numeric_features = numeric_features.to(device)
            captions = captions.to(device)

            captions_in = captions[:, :-1]
            targets = captions[:, 1:]

            _, logits = model(images, graphs, numeric_features, captions_in)
            pred = torch.argmax(logits, dim=-1)

            true_flat = targets.reshape(-1).detach().cpu().numpy()
            pred_flat = pred.view(-1).detach().cpu().numpy()

            mask = true_flat != pad_id
            all_true.extend(true_flat[mask].tolist())
            all_pred.extend(pred_flat[mask].tolist())

            if rdkit_available:
                for seq in pred.detach().cpu().numpy():
                    smiles = _decode_tokens(seq.tolist(), idx_to_char, start_id, end_id, pad_id)
                    total_smiles += 1
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            valid_count += 1

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

    if len(all_true) == 0:
        raise SystemExit('No tokens to evaluate (all were padding?)')

    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)

    token_accuracy = float((y_true == y_pred).mean())

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [l for l in labels if l != pad_id]

    precision_macro = float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))

    metrics = {
        'token_accuracy': token_accuracy,
        'token_precision_macro': precision_macro,
        'token_recall_macro': recall_macro,
        'token_f1_macro': f1_macro,
        'valid_smiles_rate': float(valid_count / max(total_smiles, 1)) if rdkit_available else None,
        'notes': 'Token-level metrics for SMILES generation using next-token teacher forcing (shifted); <pad> excluded.'
    }

    metrics_path = os.path.join(results_dir, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # "Class" distribution on test set (for this project: token-class distribution)
    try:
        token_counts = Counter(y_true.tolist())
        top = token_counts.most_common(max(args.top_tokens, 30))
        top = [(tok, cnt) for tok, cnt in top if tok != pad_id]

        labels_tok = [
            ('<start>' if tok == start_id else '<end>' if tok == end_id else idx_to_char.get(int(tok), str(tok)))
            for tok, _ in top
        ]
        counts = [cnt for _, cnt in top]

        plt.figure(figsize=(14, 6))
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), labels_tok, rotation=90)
        plt.title('Test Set Class Distribution (Token Frequencies)')
        plt.xlabel('Token')
        plt.ylabel('Count')
        plt.tight_layout()

        dist_path = os.path.join(docs_dir, 'test_class_distribution.png')
        plt.savefig(dist_path)
        plt.close()
    except Exception:
        dist_path = None

    # Confusion matrix for top-K frequent tokens (+ OTHER)
    counter = Counter(y_true.tolist())
    most_common = [tok for tok, _ in counter.most_common(args.top_tokens) if tok != pad_id]
    other_id = -1
    label_set = most_common + [other_id]

    def map_tok(t: int) -> int:
        return t if t in most_common else other_id

    y_true_small = np.array([map_tok(int(t)) for t in y_true], dtype=np.int64)
    y_pred_small = np.array([map_tok(int(t)) for t in y_pred], dtype=np.int64)

    cm = confusion_matrix(y_true_small, y_pred_small, labels=label_set)

    def pretty_label(t: int) -> str:
        if t == other_id:
            return 'OTHER'
        if t == pad_id:
            return '<pad>'
        if t == start_id:
            return '<start>'
        if t == end_id:
            return '<end>'
        return idx_to_char.get(int(t), str(t))

    tick_labels = [pretty_label(t) for t in label_set]

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
    plt.title(f'Confusion Matrix (Top {len(most_common)} tokens + OTHER)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    cm_path = os.path.join(docs_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    print('Saved:')
    print(f'- {os.path.relpath(metrics_path, base_dir)}')
    print(f'- {os.path.relpath(cm_path, base_dir)}')
    if dist_path is not None:
        print(f'- {os.path.relpath(dist_path, base_dir)}')


if __name__ == '__main__':
    main()
