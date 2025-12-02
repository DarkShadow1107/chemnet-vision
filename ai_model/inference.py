"""
ChemNet-Vision Model Inference
==============================
Load trained model and predict SMILES from molecule images.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.model import get_model
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class MoleculePredictor:
    """Predictor class for molecule recognition from images."""
    
    def __init__(self, checkpoint_path=None, vocab_path=None, device=None):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            vocab_path: Path to vocabulary JSON file
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Default paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, 'saved_models', 'checkpoint_best.pth')
        if vocab_path is None:
            vocab_path = os.path.join(base_dir, 'saved_models', 'vocab.json')
        
        # Load vocabulary
        self.vocab = self._load_vocab(vocab_path)
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab) + 10  # Buffer
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded on {self.device}")
    
    def _load_vocab(self, vocab_path):
        """Load vocabulary from JSON file."""
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                return json.load(f)
        else:
            # Default SMILES vocabulary if file not found
            chars = "CNOSPFClBrI=#@+\\/-()[]0123456789cnops"
            vocab = {char: i + 3 for i, char in enumerate(chars)}
            vocab['<pad>'] = 0
            vocab['<start>'] = 1
            vocab['<end>'] = 2
            return vocab
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint."""
        model = get_model(self.vocab_size)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using untrained model!")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _create_dummy_graph(self):
        """Create a dummy graph for molecules without SMILES."""
        x = torch.zeros(1, 9, dtype=torch.float)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    
    def _smiles_to_graph(self, smiles):
        """Convert SMILES to graph (for property prediction)."""
        if not smiles:
            return self._create_dummy_graph()
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._create_dummy_graph()
            
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
                return self._create_dummy_graph()
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
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
        except:
            return self._create_dummy_graph()
    
    def _decode_smiles(self, output_ids):
        """Decode token IDs to SMILES string."""
        smiles = []
        for idx in output_ids:
            if idx == self.vocab.get('<end>', 2):
                break
            if idx == self.vocab.get('<start>', 1) or idx == self.vocab.get('<pad>', 0):
                continue
            char = self.idx_to_char.get(idx, '')
            smiles.append(char)
        return ''.join(smiles)
    
    def _validate_smiles(self, smiles):
        """Validate SMILES string using RDKit."""
        if not smiles or not isinstance(smiles, str):
            return False, None
        try:
            # Clean the SMILES string
            smiles = str(smiles).strip()
            if len(smiles) == 0:
                return False, None
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Return canonical SMILES
                canonical = Chem.MolToSmiles(mol)
                return True, str(canonical)
            return False, None
        except Exception as e:
            print(f"SMILES validation error: {e}")
            return False, None
    
    def predict_from_image(self, image_path, max_length=150):
        """
        Predict SMILES from a molecule image.
        
        Args:
            image_path: Path to the molecule image
            max_length: Maximum SMILES sequence length
            
        Returns:
            dict with predicted SMILES, validity, and confidence
        """
        # Load and preprocess image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Create dummy graph (we'll use CNN features primarily)
        graph = self._create_dummy_graph()
        graph = Batch.from_data_list([graph]).to(self.device)
        
        # Generate SMILES using greedy decoding
        with torch.no_grad():
            # Get CNN features
            img_features = self.model.cnn(image_tensor)
            img_features = img_features.view(img_features.size(0), -1)
            img_features = self.model.cnn_fc(img_features)
            
            # Initialize with <start> token
            current_token = torch.tensor([[self.vocab.get('<start>', 1)]]).to(self.device)
            hidden = img_features.unsqueeze(0).repeat(self.model.rnn.num_layers, 1, 1)
            
            generated_ids = []
            confidences = []
            
            for _ in range(max_length):
                embeddings = self.model.rnn.embedding(current_token)
                output, hidden = self.model.rnn.gru(embeddings, hidden)
                logits = self.model.rnn.fc(output)
                
                # Get probabilities
                probs = F.softmax(logits[:, -1, :], dim=-1)
                confidence, predicted = torch.max(probs, dim=-1)
                
                token_id = predicted.item()
                generated_ids.append(token_id)
                confidences.append(confidence.item())
                
                # Stop if <end> token
                if token_id == self.vocab.get('<end>', 2):
                    break
                
                current_token = predicted.unsqueeze(0)
        
        # Decode SMILES
        predicted_smiles = self._decode_smiles(generated_ids)
        is_valid, canonical_smiles = self._validate_smiles(predicted_smiles)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'predicted_smiles': predicted_smiles,
            'canonical_smiles': canonical_smiles if is_valid else None,
            'is_valid': is_valid,
            'confidence': avg_confidence,
            'sequence_length': len(generated_ids)
        }
    
    def predict_from_pil(self, pil_image, max_length=150):
        """Predict from PIL Image object."""
        return self.predict_from_image(pil_image, max_length)


# Global predictor instance (lazy loaded)
_predictor = None

def get_predictor():
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MoleculePredictor()
    return _predictor


def predict_molecule(image_path):
    """
    Convenience function to predict molecule from image.
    
    Args:
        image_path: Path to molecule image
        
    Returns:
        dict with prediction results
    """
    predictor = get_predictor()
    return predictor.predict_from_image(image_path)


# Test function
if __name__ == "__main__":
    import sys
    
    print("ChemNet-Vision Inference Test")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MoleculePredictor()
    
    # Test with an image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nTesting with: {image_path}")
        result = predictor.predict_from_image(image_path)
        print(f"Predicted SMILES: {result['predicted_smiles']}")
        print(f"Canonical SMILES: {result['canonical_smiles']}")
        print(f"Is Valid: {result['is_valid']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        # Test with a sample image from the dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_images_dir = os.path.join(base_dir, 'data', '2d_images')
        
        if os.path.exists(test_images_dir):
            images = [f for f in os.listdir(test_images_dir) if f.endswith('.png')][:3]
            
            for img_name in images:
                img_path = os.path.join(test_images_dir, img_name)
                print(f"\nTesting: {img_name}")
                result = predictor.predict_from_image(img_path)
                print(f"  Predicted: {result['predicted_smiles'][:50]}...")
                print(f"  Valid: {result['is_valid']}, Confidence: {result['confidence']:.4f}")
        else:
            print("\nNo test images found. Provide an image path as argument.")
            print("Usage: python inference.py <image_path>")
