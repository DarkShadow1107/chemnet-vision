from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import sys
import json
import io
import base64
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

# Check if AI model is available
AI_MODEL_AVAILABLE = False
try:
    from ai_model.inference import MoleculePredictor, predict_molecule
    # Try to instantiate predictor to check if model is available
    trained_checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model.pth')
    fallback_checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'checkpoint_best.pth')
    # Prefer best checkpoint if present; fallback to Etapa 5 deliverable path
    checkpoint_path = fallback_checkpoint_path if os.path.exists(fallback_checkpoint_path) else trained_checkpoint_path

    if os.path.exists(checkpoint_path):
        AI_MODEL_AVAILABLE = True
        print(f"✓ AI Model available at: {checkpoint_path}")
    else:
        print(f"✗ AI Model checkpoint not found at: {checkpoint_path}")
except ImportError as e:
    print(f"✗ AI Model import failed: {e}")
except Exception as e:
    print(f"✗ AI Model initialization error: {e}")

print(f"Inference Mode: {INFERENCE_MODE.upper()}")
print(f"AI Model Available: {AI_MODEL_AVAILABLE}")


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

# Load molecules data
MOLECULES_DATA = []
try:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'molecules.json')
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            MOLECULES_DATA = json.load(f)
    else:
        print(f"Warning: molecules.json not found at {data_path}")
except Exception as e:
    print(f"Error loading molecules.json: {e}")

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


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').lower().strip()
    
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Check for molecule names in the message with improved matching
    found_molecule = None
    best_match_length = 0
    
    for mol in MOLECULES_DATA:
        mol_name = mol.get('Name', '')
        mol_name_lower = mol_name.lower()
        
        # Skip very short names (like single digits) to avoid false matches
        if len(mol_name) < 3:
            continue
        
        # Check if the molecule name appears as a whole word in the message
        # Use word boundary checking to avoid partial matches
        import re
        # Escape special regex characters in molecule name
        escaped_name = re.escape(mol_name_lower)
        # Match as whole word (with word boundaries or at start/end of string)
        pattern = r'(?:^|[\s,;:.\'"()\[\]{}])' + escaped_name + r'(?:$|[\s,;:.\'"()\[\]{}?!])'
        
        if re.search(pattern, message) or mol_name_lower == message:
            # Prefer longer matches (more specific molecule names)
            if len(mol_name) > best_match_length:
                best_match_length = len(mol_name)
                found_molecule = mol
    
    if found_molecule:
        name = found_molecule['Name']
        # Check for SMILES with different key capitalization
        smiles = found_molecule.get('SMILES', '') or found_molecule.get('Smiles', '') or found_molecule.get('smiles', '')
        
        # Get RAG info
        rag_info = get_rag_context(name)
        
        # Generate 2D image (base64 encoded)
        image_2d = None
        if smiles:
            image_2d = generate_2d_image(smiles)
        
        # Generate 3D structure (SDF)
        sdf_block = None
        if smiles:
            sdf_block = generate_3d_structure(smiles)

        response = {
            "role": "assistant",
            "content": f"I found information about **{name}**. Here's what I know:",
            "moleculeData": {
                "name": name,
                "smiles": smiles,
                "info": rag_info,
                "structure": sdf_block,
                "format": "sdf",
                "image2d": image_2d  # Base64 encoded 2D image from RDKit
            }
        }
        return jsonify(response)
    
    # Default response if no molecule found
    # Suggest some actual molecules from the database
    sample_molecules = []
    for mol in MOLECULES_DATA[:100]:
        name = mol.get('Name', '')
        if len(name) >= 5 and name.isalpha() == False:  # Get interesting names
            sample_molecules.append(name)
        if len(sample_molecules) >= 5:
            break
    
    if not sample_molecules:
        sample_molecules = [m.get('Name', '') for m in MOLECULES_DATA[:5]]
    
    suggestions = ", ".join([f"'{m}'" for m in sample_molecules[:3]])
    
    return jsonify({
        "role": "assistant",
        "content": f"I couldn't find a molecule matching your query. Try asking about a specific molecule by its exact name.\n\nHere are some molecules in my database: {suggestions}\n\nYou can also upload a 2D molecule image for analysis!"
    })


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
        "ai_model_available": AI_MODEL_AVAILABLE,
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
            "ai_available": AI_MODEL_AVAILABLE,
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
    
    if new_mode == 'ai' and not AI_MODEL_AVAILABLE:
        return jsonify({"error": "AI mode not available - model checkpoint not found"}), 400
    
    INFERENCE_MODE = new_mode
    return jsonify({
        "message": f"Inference mode set to '{new_mode}'",
        "mode": INFERENCE_MODE
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
        if not AI_MODEL_AVAILABLE:
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
        if AI_MODEL_AVAILABLE:
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
