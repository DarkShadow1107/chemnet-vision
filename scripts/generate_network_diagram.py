"""
Script pentru generarea diagramei arhitecturii rețelei neuronale ChemNet-Vision.
Generează o imagine PNG cu structura completă a modelului.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_network_diagram():
    """Creează diagrama arhitecturii ChemNet-Vision."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis('off')
    
    # Culori
    colors = {
        'input': '#E3F2FD',      # Light blue
        'cnn': '#BBDEFB',         # Blue
        'mlp': '#C8E6C9',         # Green
        'gnn': '#FFE0B2',         # Orange
        'fusion': '#E1BEE7',      # Purple
        'lstm': '#FFCDD2',        # Red
        'output': '#F5F5F5',      # Gray
        'arrow': '#455A64',       # Dark gray
        'title': '#1565C0',       # Dark blue
    }
    
    def draw_box(x, y, width, height, text, color, fontsize=10, bold=False):
        """Desenează o cutie cu text."""
        box = FancyBboxPatch((x, y), width, height, 
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='#333333', linewidth=2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=fontsize, 
                fontweight=weight, wrap=True)
    
    def draw_arrow(start, end, color='#455A64'):
        """Desenează o săgeată."""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # ==================== TITLU ====================
    ax.text(8, 21.5, 'ChemNet-Vision Neural Network Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold', 
            color=colors['title'])
    ax.text(8, 20.8, '15,300,290 Trainable Parameters | Custom Architecture (No Pretraining)', 
            ha='center', va='center', fontsize=11, color='#666666')
    
    # ==================== INPUT LAYER ====================
    ax.text(8, 19.8, '── INPUT LAYER ──', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')
    
    # Image input
    draw_box(1, 18.2, 3.5, 1.2, 'Image Input\n224 × 224 × 3', colors['input'], fontsize=10)
    
    # Numeric features input
    draw_box(6.25, 18.2, 3.5, 1.2, 'Numeric Features\n23 Molecular Props', colors['input'], fontsize=10)
    
    # Graph input
    draw_box(11.5, 18.2, 3.5, 1.2, 'Molecular Graph\nAtoms + Bonds', colors['input'], fontsize=10)
    
    # ==================== ENCODERS ====================
    ax.text(8, 17.2, '── ENCODERS ──', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')
    
    # CNN Encoder Box
    draw_box(0.5, 11, 4.5, 5.8, '', colors['cnn'])
    ax.text(2.75, 16.4, 'CNN Encoder', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(2.75, 15.9, '(Custom - No Pretraining)', ha='center', va='center', 
            fontsize=8, color='#666666')
    
    # CNN Details
    cnn_layers = [
        ('Conv 3→64 + BN + ReLU', 15.3),
        ('ResBlock ×2 (64→64)', 14.7),
        ('Conv 64→128 (stride=2)', 14.1),
        ('ResBlock ×2 (128→128)', 13.5),
        ('Conv 128→256 (stride=2)', 12.9),
        ('ResBlock ×2 (256→256)', 12.3),
        ('Conv 256→512 (stride=2)', 11.7),
        ('ResBlock ×2 (512→512)', 11.1),
    ]
    for text, y in cnn_layers:
        ax.text(2.75, y, text, ha='center', va='center', fontsize=7)
    
    # CNN Output
    draw_box(1, 10.2, 3.5, 0.6, 'Output: 512 dim', '#90CAF9', fontsize=9, bold=True)
    
    # MLP Encoder Box
    draw_box(5.75, 13, 4.5, 3.8, '', colors['mlp'])
    ax.text(8, 16.4, 'MLP Encoder', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    mlp_layers = [
        ('Linear: 23 → 64', 15.5),
        ('ReLU + Dropout(0.3)', 14.9),
        ('Linear: 64 → 128', 14.3),
        ('ReLU + Dropout(0.3)', 13.7),
    ]
    for text, y in mlp_layers:
        ax.text(8, y, text, ha='center', va='center', fontsize=8)
    
    # MLP Output
    draw_box(6.25, 12.2, 3.5, 0.6, 'Output: 128 dim', '#A5D6A7', fontsize=9, bold=True)
    
    # GNN Encoder Box
    draw_box(11, 13, 4.5, 3.8, '', colors['gnn'])
    ax.text(13.25, 16.4, 'GNN Encoder', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    gnn_layers = [
        ('GCNConv: atoms → 64', 15.5),
        ('GCNConv: 64 → 64', 14.9),
        ('GCNConv: 64 → 128', 14.3),
        ('Global Mean Pool', 13.7),
    ]
    for text, y in gnn_layers:
        ax.text(13.25, y, text, ha='center', va='center', fontsize=8)
    
    # GNN Output
    draw_box(11.5, 12.2, 3.5, 0.6, 'Output: 128 dim', '#FFCC80', fontsize=9, bold=True)
    
    # ==================== Arrows from Input to Encoders ====================
    draw_arrow((2.75, 18.2), (2.75, 16.8))
    draw_arrow((8, 18.2), (8, 16.8))
    draw_arrow((13.25, 18.2), (13.25, 16.8))
    
    # ==================== FUSION LAYER ====================
    ax.text(8, 11.2, '── FUSION LAYER ──', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')
    
    draw_box(4, 8.8, 8, 2, '', colors['fusion'])
    ax.text(8, 10.3, 'Multimodal Fusion', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(8, 9.7, 'Concat: [CNN:512 + MLP:128 + GNN:128] = 768 dim', 
            ha='center', va='center', fontsize=9)
    ax.text(8, 9.2, 'Linear: 768 → 256 + ReLU + Dropout(0.3)', 
            ha='center', va='center', fontsize=9)
    
    # Fusion Output
    draw_box(6, 8, 4, 0.6, 'Output: 256 dim (Unified Representation)', '#CE93D8', fontsize=9, bold=True)
    
    # ==================== Arrows to Fusion ====================
    draw_arrow((2.75, 10.2), (5, 9.8))
    draw_arrow((8, 12.2), (8, 10.8))
    draw_arrow((13.25, 12.2), (11, 9.8))
    
    # ==================== LSTM DECODER ====================
    ax.text(8, 7.2, '── LSTM DECODER ──', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')
    
    draw_box(3, 3.5, 10, 3.3, '', colors['lstm'])
    ax.text(8, 6.3, 'Autoregressive SMILES Decoder', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    lstm_details = [
        ('Token Embedding: 65 vocab → 256 dim', 5.7),
        ('LSTM: 2 layers, hidden_size=512, dropout=0.2', 5.1),
        ('Linear: 512 → 65 (vocab logits)', 4.5),
        ('Generation: <SOS> → token₁ → token₂ → ... → <EOS>', 3.9),
    ]
    for text, y in lstm_details:
        ax.text(8, y, text, ha='center', va='center', fontsize=9)
    
    # Arrow to LSTM
    draw_arrow((8, 8), (8, 6.8))
    
    # ==================== OUTPUT ====================
    ax.text(8, 2.8, '── OUTPUT ──', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#333333')
    
    draw_box(4.5, 1.5, 7, 1, 'SMILES String\nExample: "CCO" (Ethanol), "c1ccccc1" (Benzene)', 
             colors['output'], fontsize=10)
    
    # Arrow to Output
    draw_arrow((8, 3.5), (8, 2.5))
    
    # ==================== VOCABULARY BOX ====================
    draw_box(12, 4.5, 3.5, 2, 'Vocabulary\n65 Tokens\n\n<PAD> <SOS> <EOS>\nC c N n O o S\n( ) [ ] = # ...',
             '#FFF9C4', fontsize=8)
    
    # ==================== LEGEND ====================
    legend_items = [
        (colors['cnn'], 'CNN: Visual Features'),
        (colors['mlp'], 'MLP: Numeric Features'),
        (colors['gnn'], 'GNN: Graph Structure'),
        (colors['fusion'], 'Fusion: Multimodal'),
        (colors['lstm'], 'LSTM: Sequence Generation'),
    ]
    
    for i, (color, label) in enumerate(legend_items):
        y_pos = 1.8 - i * 0.35
        rect = mpatches.Rectangle((0.3, y_pos - 0.1), 0.4, 0.25, 
                                   facecolor=color, edgecolor='#333333', linewidth=1)
        ax.add_patch(rect)
        ax.text(0.85, y_pos, label, ha='left', va='center', fontsize=8)
    
    # ==================== STATISTICS BOX ====================
    draw_box(12, 0.5, 3.5, 1.8, 'Model Stats\n\nParams: 15.3M\nTrain Loss: 0.0002\nVal Loss: 0.0001',
             '#E8F5E9', fontsize=8)
    
    plt.tight_layout()
    
    # Salvare
    output_path = 'docs/network_architecture.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Diagrama salvată în: {output_path}")
    
    # Salvare și în format SVG pentru calitate mai bună
    svg_path = 'docs/network_architecture.svg'
    plt.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Diagrama SVG salvată în: {svg_path}")
    
    plt.close()
    
    return output_path

if __name__ == "__main__":
    create_network_diagram()
