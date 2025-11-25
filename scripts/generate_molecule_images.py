import json
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

def clean_filename(name: str) -> str:
    # Keep letters, digits, spaces, -, _, and dot
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.")
    return "".join(c for c in name if c in keep).strip()

def generate_representations(json_path, output_base):
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            molecules = json.load(f)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return
    
    dir_2d = os.path.join(output_base, '2d_images')
    os.makedirs(dir_2d, exist_ok=True)

    for i, mol_data in enumerate(molecules):
        name = mol_data.get('Name')
        smiles = mol_data.get('Smiles') # JSON keys might be case sensitive, check CSV conversion
        
        # Fallback if keys are different (CSV to JSON usually preserves case)
        if not name:
            name = f"molecule_{i}"
        else:
            name = str(name) # Ensure name is a string
            
        if not smiles:
            # Try uppercase SMILES if lowercase fails
            smiles = mol_data.get('SMILES')
        
        if not smiles or pd.isna(smiles): # pd.isna might not work if pd is not imported or used on dict
             if not smiles:
                # print(f"No SMILES for {name}, skipping.")
                continue
            
        safe_name = clean_filename(name)
        if not safe_name:
            safe_name = f"molecule_{i}"

        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                print(f"Invalid SMILES for {name}")
                continue
            
            # Generate 2D Image
            img_path = os.path.join(dir_2d, f"{safe_name}.png")
            if not os.path.exists(img_path):
                Draw.MolToFile(mol, img_path, size=(300, 300))
                # print(f"Generated 2D image for {name}")
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    print("Generation complete.")

if __name__ == "__main__":
    # We don't need pandas for the JSON version necessarily, but let's keep imports clean
    import pandas as pd # Just in case we need it for isna check, though we can do it manually
    
    json_file = os.path.join('data', 'molecules.json')
    output_folder = os.path.join('data')
    generate_representations(json_file, output_folder)
