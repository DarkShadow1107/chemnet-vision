# ChemNet-Vision

An AI-powered system for molecule recognition and analysis using GNN and RNN architectures.

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR

## Project Structure

```
chemnet-vision/
├── README.md
├── docs/
│   └── datasets/           # Descriere seturi de date, rapoarte EDA
├── data/
│   ├── raw/                # Date brute
│   ├── processed/          # Date curățate și transformate
│   ├── train/              # Set de instruire
│   ├── validation/         # Set de validare
│   ├── test/               # Set de testare
│   └── README.md           # Documentație dataset
├── src/
│   ├── preprocessing/      # Funcții pentru preprocesare
│   ├── app/                # Next.js Frontend
│   └── components/         # React Components
├── ai_model/               # PyTorch models (GNN + RNN)
├── backend/                # Flask backend API
├── scripts/                # Utility scripts
├── config/                 # Fișiere de configurare
└── requirements.txt        # Dependențe Python
```

## Etapa 3: Analiza și Pregătirea Setului de Date

### Preprocesare Date

```bash
python src/preprocessing/data_preprocessing.py
```

### Rezultate Preprocesare:

-   **Dataset original:** 48,960 molecule (ChEMBL)
-   **Dataset final:** 42,149 molecule cu SMILES valid
-   **Împărțire:** Train 70% | Validation 15% | Test 15%

### Pași de preprocesare aplicați:

1. ✅ Eliminarea duplicatelor
2. ✅ Validarea și filtrarea SMILES
3. ✅ Imputarea valorilor lipsă (mediană)
4. ✅ Tratarea outlierilor (IQR capping)
5. ✅ Encoding variabile categoriale
6. ✅ Extragerea descriptorilor moleculari (10 RDKit)
7. ✅ Normalizare Min-Max

## Setup

1.  **Install Dependencies:**

    ```bash
    # Python
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

    # Node.js
    npm install
    ```

2.  **Run Scripts:**

    -   Preprocesare date: `python src/preprocessing/data_preprocessing.py`
    -   Convert CSV to JSON: `python scripts/csv_to_json.py`
    -   Download Wikipedia PDFs: `python scripts/wiki_pdf_downloader.py`
    -   Generate Images: `python scripts/generate_molecule_images.py`

3.  **Train Model:**

    ```bash
    python ai_model/train.py
    ```

4.  **Run Application:**
    -   **Backend:**
        ```bash
        python backend/app.py
        ```
    -   **Frontend:**
        ```bash
        npm run dev -- --turbo
        ```

## Features

-   **AI System:** Uses Graph Neural Networks (GNN) and Recurrent Neural Networks (RNN) for molecule analysis.
-   **Data Processing:** Automated EDA, preprocessing, and train/val/test splitting.
-   **Frontend:** Next.js 15+ with React Compiler and Tailwind CSS.
-   **Backend:** Flask API.
