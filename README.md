# ChemNet-Vision

An AI-powered system for molecule recognition and analysis using GNN and RNN architectures.

## Project Structure

-   `ai_model/`: PyTorch models (GNN + RNN) and training script.
-   `backend/`: Flask backend API.
-   `data/`: Data storage for CSVs, JSONs, PDFs, and Images.
-   `scripts/`: Utility scripts for data processing.
-   `src/`: Next.js Frontend source code.

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
-   **Frontend:** Next.js 15+ with React Compiler and Tailwind CSS.
-   **Backend:** Flask API.
-   **Data Processing:** Automated scripts for data conversion, PDF downloading, and 2D/3D image generation.
