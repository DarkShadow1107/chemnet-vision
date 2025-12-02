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
    return jsonify({"message": "ChemNet-Vision Backend is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save file temporarily
    filepath = os.path.join('data', 'temp', file.filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    
    # Use trained model for prediction
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
            molecule_name = "Predicted Molecule"
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
            
            result = {
                "molecule": molecule_name,
                "smiles": predicted_smiles,
                "info": rag_info,
                "structure": sdf_block,
                "format": "sdf",
                "image2d": image_2d,
                "confidence": float(confidence),
                "is_valid": bool(is_valid)
            }
        else:
            # Invalid prediction - use fallback
            result = {
                "molecule": "Unknown (Low Confidence)",
                "smiles": str(prediction.get('predicted_smiles', '')),
                "info": f"The model could not confidently identify this molecule. Confidence: {confidence:.1%}. The predicted SMILES may be invalid.",
                "structure": None,
                "format": None,
                "image2d": None,
                "confidence": float(confidence),
                "is_valid": False
            }
        
        # Cleanup temp file
        try:
            os.remove(filepath)
        except:
            pass
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback to mock response if model fails
        molecule_name = "Aspirin (Fallback)"
        aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        
        rag_info = get_rag_context("Aspirin")
        if not rag_info or "Knowledge base not available" in rag_info:
            rag_info = "Aspirin, also known as acetylsalicylic acid, is a medication used to reduce pain, fever, or inflammation."

        # Generate 2D image using RDKit
        image_2d = generate_2d_image(aspirin_smiles)
        
        # Generate 3D structure using RDKit
        sdf_block = generate_3d_structure(aspirin_smiles)
        
        result = {
            "molecule": molecule_name, 
            "smiles": aspirin_smiles,
            "info": f"Model error: {str(e)}. Showing fallback molecule. " + rag_info,
            "structure": sdf_block,
            "format": "sdf",
            "image2d": image_2d
        }
        
        return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
