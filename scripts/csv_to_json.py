import pandas as pd
import json
import os
import sys

def convert_csv_to_json(csv_path, json_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    try:
        # The CSV uses semicolons as separators
        # on_bad_lines='warn' will skip bad lines and print a warning
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='warn')
        # Convert to list of dicts
        data = df.to_dict(orient='records')
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Successfully converted {csv_path} to {json_path}")
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")

if __name__ == "__main__":
    # Example usage
    # You can pass arguments via command line if needed
    csv_file = os.path.join('data', 'molecules.csv')
    json_file = os.path.join('data', 'molecules.json')
    
    # Create dummy CSV if it doesn't exist for testing
    if not os.path.exists(csv_file):
        print("Creating dummy CSV for testing...")
        dummy_data = {
            "Name": ["Aspirin", "Caffeine", "Water"],
            "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "O"]
        }
        pd.DataFrame(dummy_data).to_csv(csv_file, index=False)

    convert_csv_to_json(csv_file, json_file)
