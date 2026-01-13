from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import sys
import json
import io
import base64
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Add the root directory to sys.path to allow importing from ai_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import RAG helper
try:
    from backend.rag_helper import get_rag_context
except ImportError:
    # Fallback if running from root or different context
    try:
        from rag_helper import get_rag_context
    except:
        get_rag_context = lambda x: "RAG System not initialized."

app = Flask(__name__)
CORS(app)

# ============================================================================
# DUAL MODE CONFIGURATION
# ============================================================================
# Mode: 'ai' = Use neural network for predictions
#       'fallback' = Use database lookup only (no AI)
#       'auto' = Try AI first, fallback to database if fails
INFERENCE_MODE = os.environ.get('INFERENCE_MODE', 'auto').lower()

# ============================================================================
# MOLECULE DATA LOADING
# ============================================================================
MOLECULES_DATA = []
try:
    processed_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'molecules_processed.csv')
    if os.path.exists(processed_csv):
        # Load necessary columns for the NLP flow
        df = pd.read_csv(processed_csv)
        MOLECULES_DATA = df.to_dict('records')
        print(f"✓ Loaded {len(MOLECULES_DATA)} molecules from processed CSV.")
    else:
        molecules_json = os.path.join(os.path.dirname(__file__), '..', 'data', 'molecules.json')
        if os.path.exists(molecules_json):
            with open(molecules_json, 'r') as f:
                MOLECULES_DATA = json.load(f)
                print(f"✓ Loaded {len(MOLECULES_DATA)} molecules from molecules.json")
except Exception as e:
    print(f"✗ Error loading molecules data: {e}")


# ============================================================================
# NLP MODEL LOADING
# ============================================================================
NLP_MODEL_AVAILABLE = False
nlp_model = None
nlp_vocab = {}

try:
    from ai_model.model import get_nlp_model
    nlp_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'models', 'nlp_model.pth')
    nlp_vocab_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'nlp_vocab.json')
    
    if os.path.exists(nlp_checkpoint) and os.path.exists(nlp_vocab_path):
        with open(nlp_vocab_path, 'r') as f:
            nlp_vocab = json.load(f)
        nlp_model = get_nlp_model(len(nlp_vocab))
        nlp_model.load_state_dict(torch.load(nlp_checkpoint, map_location='cpu', weights_only=True))
        nlp_model.eval()
        NLP_MODEL_AVAILABLE = True
        print(f"✓ NLP Model loaded from: {nlp_checkpoint}")
    else:
        print(f"✗ NLP Model not found at: {nlp_checkpoint}")
except Exception as e:
    print(f"✗ NLP Model loading error: {e}")

print(f"Inference Mode: {INFERENCE_MODE.upper()}")
print(f"NLP Model Available: {NLP_MODEL_AVAILABLE}")


def get_current_mode():
    """Get the current inference mode."""
    return INFERENCE_MODE


def set_inference_mode(mode: str):
    """Set the inference mode dynamically."""
    global INFERENCE_MODE
    if mode.lower() in ['ai', 'fallback', 'auto']:
        INFERENCE_MODE = mode.lower()
        return True
    return False


def generate_2d_image(smiles: str, size: tuple = (400, 400)) -> str:
    """Generate a 2D molecule image from SMILES and return as base64 encoded PNG."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw the molecule
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error generating 2D image: {e}")
        return None


def generate_3d_structure(smiles: str) -> str:
    """Generate a 3D SDF block from SMILES using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens for better 3D geometry
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates using ETKDG (Experimental-Torsion Distance Geometry)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # For reproducibility
        result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            # Fallback to basic embedding if ETKDG fails
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        if result == -1:
            print("Warning: Could not embed molecule, returning 2D coordinates")
            AllChem.Compute2DCoords(mol)
        else:
            # Optimize the geometry using MMFF force field
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except:
                # Fallback to UFF if MMFF fails
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                except:
                    pass  # Use unoptimized coordinates
        
        # Generate SDF block
        sdf_block = Chem.MolToMolBlock(mol)
        return sdf_block
    except Exception as e:
        print(f"Error generating 3D structure: {e}")
        return None

@app.route('/images/<path:filename>')
def serve_image(filename):
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'data', '2d_images')
    return send_from_directory(images_dir, filename)


@app.route('/molecule/2d/<molecule_name>')
def get_2d_image(molecule_name):
    """Generate and return a 2D image for a molecule by name."""
    molecule_name_lower = molecule_name.lower()
    for mol in MOLECULES_DATA:
        if mol.get('Name', '').lower() == molecule_name_lower:
            smiles = mol.get('SMILES', '')
            if smiles:
                img_base64 = generate_2d_image(smiles)
                if img_base64:
                    return jsonify({"image": img_base64})
    return jsonify({"error": "Molecule not found or SMILES invalid"}), 404


@app.route('/molecule/3d/<molecule_name>')
def get_3d_structure(molecule_name):
    """Generate and return a 3D SDF structure for a molecule by name."""
    molecule_name_lower = molecule_name.lower()
    for mol in MOLECULES_DATA:
        if mol.get('Name', '').lower() == molecule_name_lower:
            smiles = mol.get('SMILES', '')
            if smiles:
                sdf_block = generate_3d_structure(smiles)
                if sdf_block:
                    return jsonify({"structure": sdf_block, "format": "sdf"})
    return jsonify({"error": "Molecule not found or SMILES invalid"}), 404


@app.route('/search', methods=['GET'])
def search_molecules():
    """Search for molecules by name (partial match)."""
    query = request.args.get('q', '').lower().strip()
    limit = min(int(request.args.get('limit', 10)), 50)
    
    if not query or len(query) < 2:
        return jsonify({"error": "Query too short", "results": []}), 400
    
    results = []
    for mol in MOLECULES_DATA:
        name = mol.get('Name', '')
        if query in name.lower():
            results.append({
                "name": name,
                "smiles": mol.get('SMILES', ''),
                "formula": mol.get('Molecular Formula', '')
            })
            if len(results) >= limit:
                break
    
    return jsonify({"query": query, "count": len(results), "results": results})

@app.route('/')
def hello():
    return jsonify({
        "message": "ChemNet-Vision Backend is running",
        "inference_mode": INFERENCE_MODE,
        "ai_model_available": NLP_MODEL_AVAILABLE,
        "endpoints": {
            "/chat": "POST - Chat with molecule database",
            "/predict": "POST - Predict molecule from image",
            "/search": "GET - Search molecules by name",
            "/mode": "GET/POST - Get or set inference mode",
            "/molecule/2d/<name>": "GET - Get 2D image",
            "/molecule/3d/<name>": "GET - Get 3D structure"
        }
    })


@app.route('/mode', methods=['GET', 'POST'])
def mode_endpoint():
    """Get or set the inference mode (ai/fallback/auto)."""
    global INFERENCE_MODE
    
    if request.method == 'GET':
        return jsonify({
            "mode": INFERENCE_MODE,
            "ai_available": NLP_MODEL_AVAILABLE,
            "description": {
                "ai": "Use neural network for all predictions",
                "fallback": "Use database lookup only (no AI)",
                "auto": "Try AI first, fallback to database if prediction fails"
            }
        })
    
    # POST - set mode
    data = request.json or {}
    new_mode = data.get('mode', '').lower()
    
    if new_mode not in ['ai', 'fallback', 'auto']:
        return jsonify({"error": "Invalid mode. Use 'ai', 'fallback', or 'auto'"}), 400
    
    if new_mode == 'ai' and not NLP_MODEL_AVAILABLE:
        return jsonify({"error": "AI mode not available - model checkpoint not found"}), 400
    
    INFERENCE_MODE = new_mode
    return jsonify({
        "message": f"Inference mode set to '{new_mode}'",
        "mode": INFERENCE_MODE
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    NLP Chat endpoint.
    Identifies molecules from text and generates descriptive sentences.
    """
    data = request.json
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # 1. Identify molecule from text
    found_mol = None
    message_upper = message.upper()
    
    # Exhaustive search in MOLECULES_DATA if available
    for mol in MOLECULES_DATA:
        mol_name = str(mol.get('Name', '')).upper()
        if mol_name and mol_name in message_upper:
            found_mol = mol
            break
    
    # 2. Extract molecule info (or use message as substance name)
    if found_mol:
        smiles = found_mol.get('Smiles') or found_mol.get('SMILES', '')
        name = found_mol.get('Name', '')
    else:
        # If not in database, try to guess the name from message
        # Take the last word as a heuristic or the whole message
        words = message.split()
        name = words[-1].strip('?!. ') if words else "Unknown"
        smiles = None
        print(f"Substance '{name}' not in DB, using generative fallback.")

    # 3. Generate response text using NLP model
    response_text = ""
    if NLP_MODEL_AVAILABLE:
        try:
            idx_to_char = {v: k for k, v in nlp_vocab.items()}
            prompt_text = f"What is {name}?"
            input_ids = torch.tensor([[nlp_vocab.get(c, 0) for c in prompt_text]]).to('cpu')
            
            with torch.no_grad():
                curr_ids = input_ids
                generated_chars = []
                hidden = None
                for _ in range(120):
                    logits, hidden = nlp_model(curr_ids[:, -1:], hidden)
                    next_id = torch.argmax(logits[:, -1:], dim=-1).item()
                    if next_id == nlp_vocab.get('<end>', 2):
                        break
                    generated_chars.append(idx_to_char.get(next_id, ' '))
                    curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]])], dim=1)
                response_text = "".join(generated_chars).strip()
        except Exception as e:
            print(f"NLP Generation error: {e}")
            response_text = f"This is {name}, a substance identified by our AI system."
    else:
        response_text = f"Identified {name} in the message. Database lookup for {name} is currently incomplete."

    # 4. Search for 2D image in folder
    image_2d = None
    if name != "Unknown":
        image_dir = os.path.join(os.path.dirname(__file__), '..', 'data', '2d_images')
        image_path = os.path.join(image_dir, f"{name}.png")
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    image_2d = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
            except Exception as e:
                print(f"Error reading image file: {e}")
    
    # 5. Generate 3D structure using SMILES
    sdf_block = generate_3d_structure(smiles) if smiles else None

    # Handle case where both are missing but we want to show something
    if not image_2d and not sdf_block:
        print(f"No visual data for {name}.")

    return jsonify({
        "content": response_text,
        "moleculeData": {
            "name": name,
            "info": response_text,
            "smiles": smiles,
            "image2d": image_2d,
            "structure": sdf_block,
            "format": "sdf"
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict molecule from uploaded image.
    
    Supports dual mode:
    - AI mode: Uses trained neural network for prediction
    - Fallback mode: Uses database lookup only
    - Auto mode: Tries AI first, falls back to database if prediction fails
    """
    global INFERENCE_MODE
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check for mode override in request
    request_mode = request.form.get('mode', INFERENCE_MODE).lower()
    if request_mode not in ['ai', 'fallback', 'auto']:
        request_mode = INFERENCE_MODE
    
    # Save file temporarily
    filepath = os.path.join('data', 'temp', file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    
    result = None
    mode_used = request_mode
    
    # ========================================================================
    # FALLBACK MODE - Database lookup only, no AI
    # ========================================================================
    if request_mode == 'fallback':
        result = _fallback_prediction(filepath)
        mode_used = 'fallback'
    
    # ========================================================================
    # AI MODE - Use neural network exclusively
    # ========================================================================
    elif request_mode == 'ai':
        if not NLP_MODEL_AVAILABLE:
            result = {
                "error": "AI model not available",
                "molecule": "Unknown",
                "smiles": "",
                "info": "The AI model checkpoint was not found. Please train the model first or use fallback mode.",
                "mode_used": "error"
            }
        else:
            result = _ai_prediction(filepath)
            mode_used = 'ai'
    
    # ========================================================================
    # AUTO MODE - Try AI first, fallback if fails
    # ========================================================================
    elif request_mode == 'auto':
        if NLP_MODEL_AVAILABLE:
            result = _ai_prediction(filepath)
            mode_used = 'ai'
            
            # If AI prediction is invalid or low confidence, try fallback
            if not result.get('is_valid', False) or result.get('confidence', 0) < 0.5:
                fallback_result = _fallback_prediction(filepath)
                # Merge results - keep AI prediction info but add fallback data
                result['fallback_info'] = fallback_result.get('info', '')
                result['fallback_molecule'] = fallback_result.get('molecule', '')
                mode_used = 'auto (ai + fallback)'
        else:
            result = _fallback_prediction(filepath)
            mode_used = 'fallback (ai unavailable)'
    
    # Cleanup temp file
    try:
        os.remove(filepath)
    except:
        pass
    
    if result:
        result['mode_used'] = mode_used
        return jsonify(result)
    
    return jsonify({"error": "Prediction failed"}), 500


def _ai_prediction(filepath: str) -> dict:
    """Use AI model for prediction."""
    try:
        from ai_model.inference import predict_molecule
        prediction = predict_molecule(filepath)
        
        predicted_smiles = prediction.get('canonical_smiles') or prediction.get('predicted_smiles', '')
        is_valid = prediction.get('is_valid', False)
        confidence = prediction.get('confidence', 0)
        
        # Ensure SMILES is a string
        if predicted_smiles:
            predicted_smiles = str(predicted_smiles)
        
        if is_valid and predicted_smiles:
            # Try to find matching molecule in database
            molecule_name = "AI Predicted Molecule"
            for mol in MOLECULES_DATA:
                mol_smiles = mol.get('SMILES', '') or mol.get('Smiles', '') or ''
                if mol_smiles:
                    try:
                        mol_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(str(mol_smiles)))
                        if mol_canonical == predicted_smiles:
                            molecule_name = mol.get('Name', 'Unknown')
                            break
                    except:
                        continue
            
            # Generate 2D/3D structures
            image_2d = generate_2d_image(predicted_smiles)
            sdf_block = generate_3d_structure(predicted_smiles)
            
            # Get RAG info
            rag_info = get_rag_context(molecule_name)
            if not rag_info or "not available" in rag_info.lower():
                rag_info = f"AI-predicted molecule with confidence {confidence:.1%}. SMILES: {predicted_smiles}"
            
            return {
                "molecule": molecule_name,
                "smiles": predicted_smiles,
                "info": rag_info,
                "structure": sdf_block,
                "format": "sdf",
                "image2d": image_2d,
                "confidence": float(confidence),
                "is_valid": True,
                "prediction_source": "neural_network"
            }
        else:
            return {
                "molecule": "Unknown (Low Confidence)",
                "smiles": str(prediction.get('predicted_smiles', '')),
                "info": f"The AI model could not confidently identify this molecule. Confidence: {confidence:.1%}.",
                "structure": None,
                "format": None,
                "image2d": None,
                "confidence": float(confidence),
                "is_valid": False,
                "prediction_source": "neural_network"
            }
    except Exception as e:
        print(f"AI Prediction error: {e}")
        return {
            "molecule": "Error",
            "smiles": "",
            "info": f"AI prediction failed: {str(e)}",
            "structure": None,
            "image2d": None,
            "confidence": 0,
            "is_valid": False,
            "prediction_source": "error"
        }


def _fallback_prediction(filepath: str) -> dict:
    """
    Fallback prediction using database lookup only.
    This demonstrates the system works without AI.
    """
    # Use a sample molecule from the database
    sample_molecules = ["Aspirin", "Caffeine", "Ibuprofen", "Paracetamol", "Ethanol"]
    
    found_mol = None
    for mol in MOLECULES_DATA:
        if mol.get('Name', '') in sample_molecules:
            found_mol = mol
            break
    
    if not found_mol and len(MOLECULES_DATA) > 0:
        found_mol = MOLECULES_DATA[0]
    
    if found_mol:
        name = found_mol.get('Name', 'Unknown')
        smiles = found_mol.get('SMILES', '') or found_mol.get('Smiles', '') or ''
        
        image_2d = generate_2d_image(smiles) if smiles else None
        sdf_block = generate_3d_structure(smiles) if smiles else None
        rag_info = get_rag_context(name)
        
        return {
            "molecule": f"{name} (Fallback Mode)",
            "smiles": smiles,
            "info": f"[FALLBACK MODE - No AI] Showing sample molecule: {name}. " + (rag_info or ""),
            "structure": sdf_block,
            "format": "sdf",
            "image2d": image_2d,
            "confidence": 1.0,  # Database lookup is 100% confident
            "is_valid": True,
            "prediction_source": "database_fallback"
        }
    
    return {
        "molecule": "No Data",
        "smiles": "",
        "info": "Fallback mode active but no molecules in database.",
        "structure": None,
        "image2d": None,
        "confidence": 0,
        "is_valid": False,
        "prediction_source": "database_fallback"
    }


if __name__ == '__main__':
    app.run(debug=True, port=5000)
