from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
from rdkit import Chem
from rdkit.Chem import AllChem

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

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '').lower()
    
    if not message:
        return jsonify({"error": "No message provided"}), 400

    # Check for molecule names in the message
    found_molecule = None
    for mol in MOLECULES_DATA:
        # Simple substring match
        if mol.get('Name', '').lower() in message:
            found_molecule = mol
            break
    
    if found_molecule:
        name = found_molecule['Name']
        smiles = found_molecule.get('SMILES', '')
        
        # Get RAG info
        rag_info = get_rag_context(name)
        
        # Generate 3D structure (SDF)
        sdf_block = None
        if smiles:
            try:
                mol_3d = Chem.MolFromSmiles(smiles)
                if mol_3d:
                    mol_3d = Chem.AddHs(mol_3d)
                    AllChem.EmbedMolecule(mol_3d)
                    sdf_block = Chem.MolToMolBlock(mol_3d)
            except Exception as e:
                print(f"Error generating 3D structure for {name}: {e}")

        response = {
            "role": "assistant",
            "content": f"I found information about {name}.",
            "moleculeData": {
                "name": name,
                "info": rag_info,
                "structure": sdf_block,
                "format": "sdf"
            },
            # Assuming images are named "Name.png"
            "image": f"http://localhost:5000/images/{name}.png" 
        }
        return jsonify(response)
    
    # Default response if no molecule found
    # In a real app, you might want to use an LLM here to chat generally
    return jsonify({
        "role": "assistant",
        "content": "I can help you with molecule information. Try asking about a specific molecule like 'Aspirin' or 'Caffeine'."
    })

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
    
    # TODO: Load model and predict
    # from ai_model.inference import predict_molecule
    # result = predict_molecule(filepath)
    
    # Mock response with 3D structure data (Aspirin example)
    molecule_name = "Aspirin"
    
    # Retrieve info from RAG
    rag_info = get_rag_context(molecule_name)
    if not rag_info or "Knowledge base not available" in rag_info:
        # Fallback info if RAG fails or is empty
        rag_info = "Aspirin, also known as acetylsalicylic acid, is a medication used to reduce pain, fever, or inflammation."

    aspirin_sdf = """
    Aspirin
  -OEChem-01234567892D

 13 13  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124    2.1000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.4249   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.4249   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6373    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.6373   -2.1000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.8498    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.8498   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0622    2.1000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    7.2747    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    7.2747   -1.4000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  2  4  1  0  0  0  0
  4  5  2  0  0  0  0
  4  6  1  0  0  0  0
  5  7  1  0  0  0  0
  6  8  2  0  0  0  0
  7  9  2  0  0  0  0
  8  9  1  0  0  0  0
  8 10  1  0  0  0  0
 10 11  2  0  0  0  0
 10 12  1  0  0  0  0
 12 13  1  0  0  0  0
M  END
"""
    
    result = {
        "molecule": f"{molecule_name} (Mock)", 
        "info": rag_info,
        "structure": aspirin_sdf
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
