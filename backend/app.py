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
# MOLECULE DATA LOADING & INDEXING
# ============================================================================
MOLECULES_DATA = []
MOLECULES_LOOKUP = {} # Fast search: { "NAME": record, "SYNONYM": record }

def load_molecule_data():
    global MOLECULES_DATA, MOLECULES_LOOKUP
    try:
        processed_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'molecules_processed.csv')
        if os.path.exists(processed_csv):
            df = pd.read_csv(processed_csv)
            MOLECULES_DATA = df.to_dict('records')
            
            # Create indexing for fast lookup
            for mol in MOLECULES_DATA:
                # Index by Name
                name_val = str(mol.get('Name', '')).upper().strip()
                if name_val and name_val != 'NAN':
                    MOLECULES_LOOKUP[name_val] = mol
                
                # Index by Synonyms (Handle |, ;, and ,)
                syns = str(mol.get('Synonyms', '')).upper()
                if syns and syns != 'NAN':
                    # Normalize delimiters
                    normalized_syns = syns.replace('|', ',').replace(';', ',')
                    for syn in normalized_syns.split(','):
                        syn_clean = syn.strip()
                        if len(syn_clean) > 2 and syn_clean not in MOLECULES_LOOKUP:
                            MOLECULES_LOOKUP[syn_clean] = mol
            
            print(f"✓ Indexed {len(MOLECULES_LOOKUP)} searchable terms from {len(MOLECULES_DATA)} molecules.")
        else:
            print(f"✗ Processed CSV not found at: {processed_csv}")
    except Exception as e:
        print(f"✗ Error loading molecules data: {e}")

load_molecule_data()

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "db_size": len(MOLECULES_DATA),
        "index_size": len(MOLECULES_LOOKUP),
        "nlp_available": NLP_MODEL_AVAILABLE
    })


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

    # 1. Identify molecule from text (Optimized Dictionary Lookup)
    found_mol = None
    clean_message = message.upper().strip()
    # Remove punctuation for better matching
    for char in '?!.,;':
        clean_message = clean_message.replace(char, ' ')
    clean_message = clean_message.strip()
    
    # Try direct lookup first
    if clean_message in MOLECULES_LOOKUP:
        found_mol = MOLECULES_LOOKUP[clean_message]
    else:
        # Try word-by-word if no direct match (looking for a chemical name)
        words = clean_message.split()
        # Filter out common chat words to avoid matching "DESCRIBE" or "TELL" as a molecule
        stop_words = {'DESCRIBE', 'TELL', 'WHAT', 'INFO', 'ABOUT', 'IS', 'ME', 'SEARCH', 'THE', 'OF'}
        for word in words:
            if len(word) > 2 and word not in stop_words and word in MOLECULES_LOOKUP:
                found_mol = MOLECULES_LOOKUP[word]
                break
    
    # 2. Extract molecule info
    db_facts = ""
    extracted_name = ""
    if found_mol:
        smiles = found_mol.get('Smiles') or found_mol.get('SMILES', '')
        extracted_name = found_mol.get('Name', 'Unknown')
        formula = found_mol.get('Molecular Formula') or found_mol.get('Formula', 'N/A')
        weight = found_mol.get('MolWeight_RDKit') or found_mol.get('Molecular Weight') or found_mol.get('Weight', 'N/A')
        mol_type = found_mol.get('Type', 'Small molecule')
        
        # Get 1-3 synonyms
        synonyms_raw = found_mol.get('Synonyms', '')
        synonyms_list = []
        if isinstance(synonyms_raw, str) and synonyms_raw.lower() != 'nan':
            # Split by common delimiters
            parts = synonyms_raw.replace('|', ',').replace(';', ',').split(',')
            synonyms_list = [s.strip() for s in parts if s.strip() and s.strip().upper() != extracted_name.upper()][:3]
        
        synonyms_str = ", ".join(synonyms_list) if synonyms_list else "None listed"
        
        try:
            if isinstance(weight, (int, float, str)) and str(weight) != 'nan':
                weight = f"{float(weight):.2f}"
        except:
            pass
            
        db_facts = (f"{extracted_name} is a {mol_type}. "
                   f"Formula: {formula}. Weight: {weight} g/mol. "
                   f"Synonyms: {synonyms_str}.")
    else:
        # Fallback name extraction: ignore common words
        words = [w for w in message.split() if "".join(c for c in w if c.isalnum()).upper() not in {'TELL', 'DESCRIBE', 'WHAT', 'IS', 'INFO'}]
        extracted_name = "".join(c for c in words[-1] if c.isalnum()) if words else "Unknown"
        smiles = None

    # 3. Generate response text using NLP model + RAG
    # AI generation is now the PRIMARY source for the textual answer
    response_text = ""
    rag_context = get_rag_context(extracted_name) if extracted_name != "Unknown" else ""

    # Suppress AI only for obvious non-sense
    is_nonsense = len(extracted_name) < 2 or extracted_name.lower() in ['heh', 'gshvs', 'kjggvsjdk']

    # =====================================================================
    # CONFIDENCE FILTER (Stage 6 Optimization - Caffeine Hallucination Fix)
    # For known DB molecules, ALWAYS use factual data. AI text is secondary.
    # For unknown molecules, reject AI output below confidence threshold 0.85.
    # =====================================================================
    CONFIDENCE_THRESHOLD = 0.85

    if NLP_MODEL_AVAILABLE and not is_nonsense:
        try:
            ai_text = ""
            ai_confidence = 0.0
            idx_to_char = {v: k for k, v in nlp_vocab.items()}
            # Condition the model: Describe [Name] -> [Answer]
            # This forces the generator to focus on the target molecule
            prompt_text = f"Describe {extracted_name} -> "
            input_ids = torch.tensor([[nlp_vocab.get(c, 0) for c in prompt_text]]).to('cpu')

            with torch.no_grad():
                generated_chars = []
                char_confidences = []
                temperature = 0.3

                # First: encode the FULL prompt through LSTM to build context
                # This is critical -- the hidden state must "see" the molecule name
                logits, hidden = nlp_model(input_ids)
                # The last logit predicts the first generated character
                logits = logits[:, -1:] / temperature
                probs = torch.softmax(logits, dim=-1)
                max_prob = probs[0].max().item()
                char_confidences.append(max_prob)
                next_id = torch.argmax(probs[0], dim=-1).item()

                char = idx_to_char.get(next_id, ' ')
                if char not in ('\n', '<'):
                    generated_chars.append(char)

                # Now generate character-by-character using the hidden state
                for _ in range(250):
                    curr_token = torch.tensor([[next_id]]).to('cpu')
                    logits, hidden = nlp_model(curr_token, hidden)
                    logits = logits[:, -1:] / temperature
                    probs = torch.softmax(logits, dim=-1)
                    max_prob = probs[0].max().item()
                    char_confidences.append(max_prob)
                    next_id = torch.argmax(probs[0], dim=-1).item()

                    char = idx_to_char.get(next_id, ' ')
                    if char == '\n' or char == '<':
                        break
                    generated_chars.append(char)

                ai_text = "".join(generated_chars).strip()
                ai_confidence = sum(char_confidences) / max(len(char_confidences), 1)

            # When DB data is available, use ONLY factual data (no AI text)
            # When DB data is NOT available, use AI-generated text
            if db_facts:
                response_text = db_facts
            elif ai_text and len(ai_text) > 10 and ai_confidence >= CONFIDENCE_THRESHOLD:
                if rag_context and "not initialized" not in rag_context:
                    response_text = ai_text
                else:
                    response_text = ai_text
            elif rag_context and "not initialized" not in rag_context:
                response_text = rag_context
            else:
                response_text = f"I couldn't find detailed information about {extracted_name}. Please try a more specific molecule name."
                
        except Exception as e:
            print(f"NLP Generation error: {e}")
            response_text = f"The AI model identified {extracted_name} but had trouble generating the description."
    else:
        # Fallback if no AI or nonsense
        if is_nonsense:
            response_text = "I'm sorry, I didn't recognize that substance. Try a specific molecule like 'Caffeine' or 'Aspirin'."
        elif db_facts:
            response_text = db_facts
        else:
            response_text = f"I couldn't find detailed information about {extracted_name}. Please try a more specific molecule name."

    # 4. Search for visuals in the database (2D/3D)
    image_2d = None
    # Use the database name for file lookups
    visual_name = found_mol.get('Name', extracted_name) if found_mol else extracted_name
    
    if visual_name != "Unknown":
        image_dir = os.path.join(os.path.dirname(__file__), '..', 'data', '2d_images')
        image_path = os.path.join(image_dir, f"{visual_name}.png")
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    image_2d = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
            except Exception as e:
                print(f"Error reading image file: {e}")
    
    # 5. Generate 3D structure using SMILES
    sdf_block = generate_3d_structure(smiles) if smiles else None

    return jsonify({
        "content": "" if found_mol else response_text,
        "moleculeData": {
            "name": visual_name,
            "info": response_text,
            "smiles": smiles,
            "image2d": image_2d,
            "structure": sdf_block,
            "format": "sdf"
        } if found_mol or smiles else None
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
