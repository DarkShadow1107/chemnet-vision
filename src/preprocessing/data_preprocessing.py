"""
Etapa 3: Preprocesarea Setului de Date pentru Rețele Neuronale
Proiect: ChemNet Vision - Analiza Moleculelor
Disciplina: Rețele Neuronale - POLITEHNICA București

Acest script realizează:
1. Analiza Exploratorie a Datelor (EDA)
2. Curățarea și Preprocesarea Datelor
3. Împărțirea în seturi train/validation/test
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
warnings.filterwarnings('ignore')

# Configurare căi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation')
TEST_DIR = os.path.join(DATA_DIR, 'test')
IMAGES_DIR = os.path.join(DATA_DIR, '2d_images')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
DOCS_DIR = os.path.join(BASE_DIR, 'docs', 'datasets')


def load_raw_data():
    """Încarcă datele brute din fișierul CSV."""
    csv_path = os.path.join(DATA_DIR, 'molecules.csv')
    
    # Citește cu separator punct-virgulă, gestionând linii problematice
    df = pd.read_csv(csv_path, sep=';', quotechar='"', on_bad_lines='skip')
    
    print(f"✅ Date încărcate: {len(df)} molecule")
    print(f"📊 Număr caracteristici: {len(df.columns)}")
    
    return df


def exploratory_data_analysis(df):
    """
    Realizează Analiza Exploratorie a Datelor (EDA).
    Returnează un dicționar cu statistici și probleme identificate.
    """
    print("\n" + "="*60)
    print("📊 ANALIZA EXPLORATORIE A DATELOR (EDA)")
    print("="*60)
    
    eda_report = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'features': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': {},
        'statistics': {},
        'problems': []
    }
    
    # 1. Analiza valorilor lipsă
    print("\n📌 1. Analiza valorilor lipsă:")
    print("-" * 40)
    
    missing = df.isnull().sum() + (df == '').sum() + (df == 'None').sum()
    missing_pct = (missing / len(df)) * 100
    
    for col in df.columns:
        pct = missing_pct[col]
        if pct > 0:
            eda_report['missing_values'][col] = {
                'count': int(missing[col]),
                'percentage': round(pct, 2)
            }
            print(f"  {col}: {missing[col]} ({pct:.2f}%)")
            
            if pct > 30:
                eda_report['problems'].append(f"Feature '{col}' are {pct:.1f}% valori lipsă (>30%)")
    
    # 2. Identificarea coloanelor numerice
    numeric_cols = ['Molecular Weight', 'Targets', 'Bioactivities', 'AlogP', 
                   'Polar Surface Area', 'HBA', 'HBD', '#RO5 Violations',
                   '#Rotatable Bonds', 'QED Weighted', 'Aromatic Rings',
                   'Heavy Atoms', 'Np Likeness Score']
    
    print("\n📌 2. Statistici descriptive pentru caracteristici numerice:")
    print("-" * 40)
    
    for col in numeric_cols:
        if col in df.columns:
            # Convertește la numeric
            series = pd.to_numeric(df[col], errors='coerce')
            valid = series.dropna()
            
            if len(valid) > 0:
                stats = {
                    'min': float(valid.min()),
                    'max': float(valid.max()),
                    'mean': float(valid.mean()),
                    'median': float(valid.median()),
                    'std': float(valid.std()),
                    'q1': float(valid.quantile(0.25)),
                    'q3': float(valid.quantile(0.75))
                }
                stats['iqr'] = stats['q3'] - stats['q1']
                eda_report['statistics'][col] = stats
                
                print(f"\n  {col}:")
                print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                print(f"    Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"    Median: {stats['median']:.2f}")
                
                # Detectarea outlierilor folosind IQR
                lower_bound = stats['q1'] - 1.5 * stats['iqr']
                upper_bound = stats['q3'] + 1.5 * stats['iqr']
                outliers = ((valid < lower_bound) | (valid > upper_bound)).sum()
                
                if outliers > 0:
                    outlier_pct = (outliers / len(valid)) * 100
                    print(f"    ⚠️ Outlieri: {outliers} ({outlier_pct:.1f}%)")
                    if outlier_pct > 5:
                        eda_report['problems'].append(
                            f"Feature '{col}' are {outlier_pct:.1f}% outlieri"
                        )
    
    # 3. Analiza caracteristicilor categoriale
    print("\n📌 3. Caracteristici categoriale:")
    print("-" * 40)
    
    categorical_cols = ['Type', 'Max Phase', 'Structure Type', 'Inorganic Flag',
                       'Passes Ro3', 'Withdrawn Flag', 'Orphan']
    
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"  {col}: {unique_vals} valori unice")
            
            # Verifică dezechilibrul claselor
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) >= 2:
                max_pct = value_counts.iloc[0] * 100
                if max_pct > 90:
                    eda_report['problems'].append(
                        f"Feature '{col}' are dezechilibru de clasă ({max_pct:.1f}% pentru clasa majoritară)"
                    )
    
    # 4. Verificarea SMILES
    print("\n📌 4. Validarea structurilor SMILES:")
    print("-" * 40)
    
    valid_smiles = 0
    invalid_smiles = 0
    
    for smiles in df['Smiles']:
        if pd.isna(smiles) or smiles == '' or smiles == 'None':
            invalid_smiles += 1
        else:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                valid_smiles += 1
            else:
                invalid_smiles += 1
    
    print(f"  ✅ SMILES valide: {valid_smiles} ({valid_smiles/len(df)*100:.1f}%)")
    print(f"  ❌ SMILES invalide: {invalid_smiles} ({invalid_smiles/len(df)*100:.1f}%)")
    
    if invalid_smiles > 0:
        eda_report['problems'].append(
            f"{invalid_smiles} molecule au SMILES invalide sau lipsă"
        )
    
    eda_report['valid_smiles'] = valid_smiles
    eda_report['invalid_smiles'] = invalid_smiles
    
    # 5. Rezumat probleme identificate
    print("\n📌 5. Probleme identificate:")
    print("-" * 40)
    
    if len(eda_report['problems']) == 0:
        print("  ✅ Nu au fost identificate probleme majore")
    else:
        for i, problem in enumerate(eda_report['problems'], 1):
            print(f"  {i}. ⚠️ {problem}")
    
    return eda_report


def preprocess_data(df, eda_report):
    """
    Curăță și preprocesează datele.
    
    Etape:
    1. Eliminare duplicate
    2. Tratarea valorilor lipsă
    3. Validare și filtrare SMILES
    4. Normalizare caracteristici numerice
    5. Encoding variabile categoriale
    6. Extragere descriptori moleculari
    """
    print("\n" + "="*60)
    print("🔧 PREPROCESAREA DATELOR")
    print("="*60)
    
    df_processed = df.copy()
    preprocessing_log = {
        'original_samples': len(df),
        'steps': []
    }
    
    # 1. Eliminarea duplicatelor
    print("\n📌 1. Eliminarea duplicatelor...")
    initial_count = len(df_processed)
    df_processed = df_processed.drop_duplicates(subset=['ChEMBL ID'])
    removed = initial_count - len(df_processed)
    print(f"  Eliminate {removed} duplicate")
    preprocessing_log['steps'].append({
        'step': 'remove_duplicates',
        'removed': removed
    })
    
    # 2. Filtrarea moleculelor cu SMILES valid
    print("\n📌 2. Validarea și filtrarea SMILES...")
    valid_mask = []
    for smiles in df_processed['Smiles']:
        if pd.isna(smiles) or smiles == '' or smiles == 'None':
            valid_mask.append(False)
        else:
            mol = Chem.MolFromSmiles(str(smiles))
            valid_mask.append(mol is not None)
    
    df_processed = df_processed[valid_mask]
    print(f"  Păstrate {len(df_processed)} molecule cu SMILES valid")
    preprocessing_log['steps'].append({
        'step': 'filter_valid_smiles',
        'remaining': len(df_processed)
    })
    
    # 3. Tratarea valorilor lipsă pentru coloane numerice
    print("\n📌 3. Tratarea valorilor lipsă...")
    
    numeric_cols = ['Molecular Weight', 'Targets', 'Bioactivities', 'AlogP', 
                   'Polar Surface Area', 'HBA', 'HBD', '#RO5 Violations',
                   '#Rotatable Bonds', 'QED Weighted', 'Aromatic Rings',
                   'Heavy Atoms', 'Np Likeness Score']
    
    imputation_stats = {}
    
    for col in numeric_cols:
        if col in df_processed.columns:
            # Convertește la numeric
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
            # Calculează median din date valide
            median_val = df_processed[col].median()
            missing_before = df_processed[col].isna().sum()
            
            # Imputare cu median
            df_processed[col] = df_processed[col].fillna(median_val)
            
            if missing_before > 0:
                imputation_stats[col] = {
                    'method': 'median',
                    'value': float(median_val),
                    'imputed_count': int(missing_before)
                }
                print(f"  {col}: {missing_before} valori imputate cu median ({median_val:.2f})")
    
    preprocessing_log['steps'].append({
        'step': 'impute_missing',
        'imputation_stats': imputation_stats
    })
    
    # 4. Tratarea outlierilor (folosind IQR capping)
    print("\n📌 4. Tratarea outlierilor (IQR capping)...")
    
    outlier_treatment = {}
    
    for col in numeric_cols:
        if col in df_processed.columns:
            q1 = df_processed[col].quantile(0.25)
            q3 = df_processed[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_before = ((df_processed[col] < lower_bound) | 
                              (df_processed[col] > upper_bound)).sum()
            
            if outliers_before > 0:
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                outlier_treatment[col] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'capped_count': int(outliers_before)
                }
                print(f"  {col}: {outliers_before} outlieri tratați")
    
    preprocessing_log['steps'].append({
        'step': 'outlier_treatment',
        'treatment': outlier_treatment
    })
    
    # 5. Encoding variabile categoriale
    print("\n📌 5. Encoding variabile categoriale...")
    
    encoding_maps = {}
    
    # Type (tipul moleculei)
    if 'Type' in df_processed.columns:
        le = LabelEncoder()
        df_processed['Type_encoded'] = le.fit_transform(df_processed['Type'].astype(str))
        encoding_maps['Type'] = dict(zip(le.classes_, range(len(le.classes_))))
        print(f"  Type: {len(le.classes_)} clase encoded")
    
    # Passes Ro3 (Regula lui 3)
    if 'Passes Ro3' in df_processed.columns:
        df_processed['Passes_Ro3_encoded'] = df_processed['Passes Ro3'].map({'Y': 1, 'N': 0})
        df_processed['Passes_Ro3_encoded'] = df_processed['Passes_Ro3_encoded'].fillna(0).astype(int)
        encoding_maps['Passes Ro3'] = {'Y': 1, 'N': 0}
        print(f"  Passes Ro3: binary encoded")
    
    # Structure Type
    if 'Structure Type' in df_processed.columns:
        le = LabelEncoder()
        df_processed['Structure_Type_encoded'] = le.fit_transform(
            df_processed['Structure Type'].astype(str)
        )
        encoding_maps['Structure Type'] = dict(zip(le.classes_, range(len(le.classes_))))
        print(f"  Structure Type: {len(le.classes_)} clase encoded")
    
    preprocessing_log['steps'].append({
        'step': 'categorical_encoding',
        'encoding_maps': encoding_maps
    })
    
    # 6. Extragerea descriptorilor moleculari din SMILES
    print("\n📌 6. Extragerea descriptorilor moleculari...")
    
    molecular_features = []
    
    for smiles in df_processed['Smiles']:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            features = {
                'MolWeight_RDKit': Descriptors.MolWt(mol),
                'LogP_RDKit': Descriptors.MolLogP(mol),
                'TPSA_RDKit': Descriptors.TPSA(mol),
                'NumHDonors_RDKit': Descriptors.NumHDonors(mol),
                'NumHAcceptors_RDKit': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds_RDKit': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings_RDKit': Descriptors.NumAromaticRings(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'RingCount': Descriptors.RingCount(mol)
            }
        else:
            features = {k: 0 for k in ['MolWeight_RDKit', 'LogP_RDKit', 'TPSA_RDKit',
                                       'NumHDonors_RDKit', 'NumHAcceptors_RDKit',
                                       'NumRotatableBonds_RDKit', 'NumAromaticRings_RDKit',
                                       'FractionCSP3', 'NumHeteroatoms', 'RingCount']}
        molecular_features.append(features)
    
    mol_df = pd.DataFrame(molecular_features)
    df_processed = pd.concat([df_processed.reset_index(drop=True), mol_df], axis=1)
    
    print(f"  Adăugați 10 descriptori moleculari RDKit")
    
    preprocessing_log['steps'].append({
        'step': 'molecular_descriptors',
        'features_added': list(mol_df.columns)
    })
    
    # 7. Normalizarea caracteristicilor pentru ML
    print("\n📌 7. Normalizarea caracteristicilor...")
    
    features_to_normalize = numeric_cols + list(mol_df.columns)
    features_to_normalize = [f for f in features_to_normalize if f in df_processed.columns]
    
    # Calculăm și salvăm parametrii de normalizare
    normalization_params = {}
    
    for col in features_to_normalize:
        min_val = df_processed[col].min()
        max_val = df_processed[col].max()
        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()
        
        normalization_params[col] = {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'std': float(std_val)
        }
        
        # Normalizare Min-Max
        if max_val > min_val:
            df_processed[f'{col}_normalized'] = (df_processed[col] - min_val) / (max_val - min_val)
        else:
            df_processed[f'{col}_normalized'] = 0
    
    print(f"  Normalizate {len(features_to_normalize)} caracteristici (Min-Max)")
    
    preprocessing_log['steps'].append({
        'step': 'normalization',
        'method': 'min-max',
        'params': normalization_params
    })
    
    # 8. Corelarea cu imaginile 2D
    print("\n📌 8. Corelarea cu imaginile 2D...")
    
    # Construim calea către imagine pentru fiecare moleculă
    def get_image_path(name):
        if pd.isna(name) or name == '' or name == 'None':
            return None
        # Numele imaginii = Name.upper() + .png
        image_name = f"{str(name).upper()}.png"
        image_path = os.path.join(IMAGES_DIR, image_name)
        if os.path.exists(image_path):
            return image_path
        return None
    
    df_processed['image_path'] = df_processed['Name'].apply(get_image_path)
    
    # Statistici despre imagini
    images_found = df_processed['image_path'].notna().sum()
    images_missing = df_processed['image_path'].isna().sum()
    
    print(f"  Imagini găsite: {images_found} ({images_found/len(df_processed)*100:.1f}%)")
    print(f"  Imagini lipsă: {images_missing} ({images_missing/len(df_processed)*100:.1f}%)")
    
    # Flag pentru a indica dacă molecula are imagine
    df_processed['has_image'] = df_processed['image_path'].notna().astype(int)
    
    preprocessing_log['steps'].append({
        'step': 'image_correlation',
        'images_found': int(images_found),
        'images_missing': int(images_missing),
        'coverage_pct': round(images_found/len(df_processed)*100, 2)
    })
    
    preprocessing_log['final_samples'] = len(df_processed)
    preprocessing_log['final_features'] = len(df_processed.columns)
    
    return df_processed, preprocessing_log, normalization_params


def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Împarte datele în seturi train/validation/test.
    Train: 70%, Validation: 15%, Test: 15%
    """
    print("\n" + "="*60)
    print("📊 ÎMPĂRȚIREA SETULUI DE DATE")
    print("="*60)
    
    # Prima împărțire: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # A doua împărțire: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio, 
        random_state=random_state
    )
    
    split_info = {
        'train_size': len(train_df),
        'validation_size': len(val_df),
        'test_size': len(test_df),
        'train_pct': len(train_df) / len(df) * 100,
        'validation_pct': len(val_df) / len(df) * 100,
        'test_pct': len(test_df) / len(df) * 100,
        'random_state': random_state
    }
    
    print(f"\n📌 Distribuția seturilor:")
    print(f"  Train:      {split_info['train_size']} ({split_info['train_pct']:.1f}%)")
    print(f"  Validation: {split_info['validation_size']} ({split_info['validation_pct']:.1f}%)")
    print(f"  Test:       {split_info['test_size']} ({split_info['test_pct']:.1f}%)")
    
    return train_df, val_df, test_df, split_info


def save_processed_data(df_processed, train_df, val_df, test_df, 
                         eda_report, preprocessing_log, split_info,
                         normalization_params):
    """Salvează toate datele și configurațiile."""
    print("\n" + "="*60)
    print("💾 SALVAREA REZULTATELOR")
    print("="*60)
    
    # 1. Salvare date brute (copie)
    raw_path = os.path.join(RAW_DIR, 'molecules_raw.csv')
    original_df = pd.read_csv(os.path.join(DATA_DIR, 'molecules.csv'), sep=';', on_bad_lines='skip')
    original_df.to_csv(raw_path, index=False)
    print(f"  ✅ Date brute: {raw_path}")
    
    # 2. Salvare date preprocesate
    processed_path = os.path.join(PROCESSED_DIR, 'molecules_processed.csv')
    df_processed.to_csv(processed_path, index=False)
    print(f"  ✅ Date preprocesate: {processed_path}")
    
    # 3. Salvare seturi train/val/test
    train_path = os.path.join(TRAIN_DIR, 'train.csv')
    val_path = os.path.join(VAL_DIR, 'validation.csv')
    test_path = os.path.join(TEST_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✅ Set train: {train_path}")
    print(f"  ✅ Set validation: {val_path}")
    print(f"  ✅ Set test: {test_path}")
    
    # 4. Salvare raport EDA
    eda_path = os.path.join(DOCS_DIR, 'eda_report.json')
    with open(eda_path, 'w', encoding='utf-8') as f:
        json.dump(eda_report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Raport EDA: {eda_path}")
    
    # 5. Salvare log preprocesare
    log_path = os.path.join(DOCS_DIR, 'preprocessing_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_log, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Log preprocesare: {log_path}")
    
    # 6. Salvare configurație preprocesare
    config = {
        'normalization_params': normalization_params,
        'split_info': split_info,
        'feature_columns': {
            'numeric': ['Molecular Weight', 'Targets', 'Bioactivities', 'AlogP', 
                       'Polar Surface Area', 'HBA', 'HBD', '#RO5 Violations',
                       '#Rotatable Bonds', 'QED Weighted', 'Aromatic Rings',
                       'Heavy Atoms', 'Np Likeness Score'],
            'categorical': ['Type', 'Structure Type', 'Passes Ro3'],
            'molecular_descriptors': ['MolWeight_RDKit', 'LogP_RDKit', 'TPSA_RDKit',
                                      'NumHDonors_RDKit', 'NumHAcceptors_RDKit',
                                      'NumRotatableBonds_RDKit', 'NumAromaticRings_RDKit',
                                      'FractionCSP3', 'NumHeteroatoms', 'RingCount'],
            'identifier': 'ChEMBL ID',
            'smiles': 'Smiles',
            'image_path': 'image_path',
            'has_image': 'has_image'
        },
        'images_dir': IMAGES_DIR
    }
    
    config_path = os.path.join(CONFIG_DIR, 'preprocessing_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Configurație: {config_path}")
    
    # 7. Salvare caracteristici numerice pentru ML (format NumPy)
    normalized_cols = [col for col in df_processed.columns if col.endswith('_normalized')]
    
    X_train = train_df[normalized_cols].values
    X_val = val_df[normalized_cols].values
    X_test = test_df[normalized_cols].values
    
    np.save(os.path.join(TRAIN_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(VAL_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(TEST_DIR, 'X_test.npy'), X_test)
    
    print(f"  ✅ Caracteristici ML salvate în format NumPy")
    print(f"     X_train shape: {X_train.shape}")
    print(f"     X_val shape: {X_val.shape}")
    print(f"     X_test shape: {X_test.shape}")
    
    # 8. Salvare liste imagini pentru fiecare set (pentru training cu imagini)
    def save_image_list(df, set_name, output_dir):
        """Salvează lista căilor către imagini pentru un set."""
        images_with_path = df[df['image_path'].notna()]['image_path'].tolist()
        
        # Salvare ca JSON
        list_path = os.path.join(output_dir, f'{set_name}_images.json')
        with open(list_path, 'w', encoding='utf-8') as f:
            json.dump({
                'count': len(images_with_path),
                'images': images_with_path
            }, f, indent=2, ensure_ascii=False)
        
        return len(images_with_path)
    
    train_images = save_image_list(train_df, 'train', TRAIN_DIR)
    val_images = save_image_list(val_df, 'validation', VAL_DIR)
    test_images = save_image_list(test_df, 'test', TEST_DIR)
    
    print(f"\n  ✅ Liste imagini salvate:")
    print(f"     Train: {train_images} imagini")
    print(f"     Validation: {val_images} imagini")
    print(f"     Test: {test_images} imagini")


def create_data_readme(eda_report, preprocessing_log, split_info):
    """Generează documentația dataset-ului."""
    
    readme_content = f"""# 📊 Documentația Setului de Date - ChemNet Vision

## Descrierea Setului de Date

### Sursa datelor
* **Origine:** ChEMBL Database - Date despre molecule și compuși chimici
* **Modul de achiziție:** Fișier extern (CSV)
* **Format original:** CSV cu separator punct-virgulă

### Caracteristicile dataset-ului original
* **Număr total de observații:** {eda_report['n_samples']}
* **Număr de caracteristici:** {eda_report['n_features']}
* **Tipuri de date:** Numerice, Categoriale, Text (SMILES)
* **Format fișiere:** CSV

---

## Caracteristici Principale

| Caracteristică | Tip | Descriere | Domeniu valori |
|----------------|-----|-----------|----------------|
| ChEMBL ID | text | Identificator unic moleculă | - |
| Name | text | Numele moleculei | - |
| Molecular Weight | numeric | Masa moleculară (Da) | 0-2500 |
| AlogP | numeric | Coeficient de partiție | -10 - 10 |
| Polar Surface Area | numeric | Suprafața polară (Å²) | 0-500 |
| HBA | numeric | Acceptori de hidrogen | 0-30 |
| HBD | numeric | Donori de hidrogen | 0-15 |
| #RO5 Violations | numeric | Încălcări regula lui 5 | 0-5 |
| Aromatic Rings | numeric | Inele aromatice | 0-10 |
| Smiles | text | Reprezentare SMILES | - |

---

## Analiza Calității Datelor

### Valori lipsă identificate
"""
    
    if eda_report['missing_values']:
        for col, info in eda_report['missing_values'].items():
            readme_content += f"* **{col}:** {info['count']} ({info['percentage']:.1f}%)\n"
    else:
        readme_content += "* Nu au fost identificate valori lipsă semnificative\n"
    
    readme_content += f"""
### Probleme identificate
"""
    
    if eda_report['problems']:
        for problem in eda_report['problems']:
            readme_content += f"* ⚠️ {problem}\n"
    else:
        readme_content += "* ✅ Nu au fost identificate probleme majore\n"
    
    readme_content += f"""
---

## Preprocesarea Datelor

### Etape aplicate:
1. **Eliminarea duplicatelor** - Pe baza ChEMBL ID
2. **Validarea SMILES** - Filtrarea moleculelor cu structuri invalide
3. **Imputarea valorilor lipsă** - Metoda: mediană
4. **Tratarea outlierilor** - IQR capping (1.5 × IQR)
5. **Encoding categorial** - LabelEncoder pentru variabile categoriale
6. **Extragerea descriptorilor moleculari** - 10 descriptori RDKit
7. **Normalizare** - Min-Max scaling

### Rezultat preprocesare:
* **Observații inițiale:** {preprocessing_log['original_samples']}
* **Observații finale:** {preprocessing_log['final_samples']}
* **Caracteristici finale:** {preprocessing_log['final_features']}

---

## Împărțirea Seturilor de Date

| Set | Număr probe | Procent |
|-----|-------------|---------|
| Train | {split_info['train_size']} | {split_info['train_pct']:.1f}% |
| Validation | {split_info['validation_size']} | {split_info['validation_pct']:.1f}% |
| Test | {split_info['test_size']} | {split_info['test_pct']:.1f}% |

**Random state:** {split_info['random_state']}

---

## Structura Fișierelor

```
data/
├── raw/
│   └── molecules_raw.csv          # Date originale
├── processed/
│   └── molecules_processed.csv    # Date preprocesate
├── train/
│   ├── train.csv                  # Set de instruire
│   └── X_train.npy                # Caracteristici normalizate
├── validation/
│   ├── validation.csv             # Set de validare
│   └── X_val.npy                  # Caracteristici normalizate
├── test/
│   ├── test.csv                   # Set de testare
│   └── X_test.npy                 # Caracteristici normalizate
└── README.md                      # Această documentație
```

---

## Descriptori Moleculari Extrași

| Descriptor | Descriere |
|------------|-----------|
| MolWeight_RDKit | Masă moleculară calculată cu RDKit |
| LogP_RDKit | Coeficient de partiție calculat |
| TPSA_RDKit | Suprafața polară topologică |
| NumHDonors_RDKit | Număr donori de hidrogen |
| NumHAcceptors_RDKit | Număr acceptori de hidrogen |
| NumRotatableBonds_RDKit | Număr legături rotabile |
| NumAromaticRings_RDKit | Număr inele aromatice |
| FractionCSP3 | Fracțiunea carbonilor sp3 |
| NumHeteroatoms | Număr heteroatomi |
| RingCount | Număr total de inele |

---

*Generat automat de scriptul de preprocesare - Etapa 3*
*Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Salvare README
    readme_path = os.path.join(DATA_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n  ✅ Documentație dataset: {readme_path}")
    
    return readme_content


def main():
    """Funcția principală care orchestrează procesul de preprocesare."""
    print("\n" + "="*60)
    print("🧪 ChemNet Vision - Preprocesarea Datelor pentru RN")
    print("   Etapa 3: Analiza și Pregătirea Setului de Date")
    print("="*60)
    
    # 1. Încărcare date
    df = load_raw_data()
    
    # 2. Analiza Exploratorie
    eda_report = exploratory_data_analysis(df)
    
    # 3. Preprocesare
    df_processed, preprocessing_log, normalization_params = preprocess_data(df, eda_report)
    
    # 4. Împărțire în seturi
    train_df, val_df, test_df, split_info = split_dataset(df_processed)
    
    # 5. Salvare rezultate
    save_processed_data(
        df_processed, train_df, val_df, test_df,
        eda_report, preprocessing_log, split_info,
        normalization_params
    )
    
    # 6. Creare documentație
    create_data_readme(eda_report, preprocessing_log, split_info)
    
    print("\n" + "="*60)
    print("✅ PREPROCESAREA COMPLETĂ!")
    print("="*60)
    print("\n📁 Fișiere generate:")
    print("   - data/raw/molecules_raw.csv")
    print("   - data/processed/molecules_processed.csv")
    print("   - data/train/train.csv, X_train.npy, train_images.json")
    print("   - data/validation/validation.csv, X_val.npy, validation_images.json")
    print("   - data/test/test.csv, X_test.npy, test_images.json")
    print("   - docs/datasets/eda_report.json")
    print("   - docs/datasets/preprocessing_log.json")
    print("   - config/preprocessing_config.json")
    print("   - data/README.md")
    print("\n🖼️ Imaginile 2D sunt corelate și pot fi folosite pentru training!")
    print("\n")


if __name__ == "__main__":
    main()
