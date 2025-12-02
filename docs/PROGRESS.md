# PROGRES SINTETIC — ChemNet Vision

**Data actualizare:** 2025-11-25

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Alexandru Gabriel  
**Data:** 2025-11-25

[![Repo](https://img.shields.io/badge/repo-DarkShadow1107/chemnet--vision-blue)](https://github.com/DarkShadow1107/chemnet-vision) [![Status](https://img.shields.io/badge/status-Stage%203%20complete-brightgreen)](https://github.com/DarkShadow1107/chemnet-vision) [![Dataset](https://img.shields.io/badge/dataset-processed-orange)](https://github.com/DarkShadow1107/chemnet-vision/tree/main/data) [![Images](https://img.shields.io/badge/images-prepared-9cf)](https://github.com/DarkShadow1107/chemnet-vision/tree/main/data/2d_images)

## 1. Rezumat executiv

-   Scop: Pregătirea unui set de date molecular și generarea de resurse conexe (imagine 2D, embeddings, index FAISS) pentru dezvoltarea și antrenarea unui sistem AI (GNN/RNN și fluxuri multimodale).
-   Stare: Etapa 3 — Analiza și pregătirea setului de date — finalizată.

Statistici cheie:

-   Dataset inițial: 48,960 molecule (ChEMBL)
-   Dataset final (SMILES valide): 42,149 molecule
-   Împărțire: Train 29,503 | Validation 6,323 | Test 6,323
-   Imagini 2D disponibile și corelate: 40,018 din 42,037 (acoperire ≈ 94.9%)

## 2. Fișiere cheie generate

-   `data/processed/molecules_processed.csv` — tabel preprocesat și îmbogățit cu descriptori și căi de imagini
-   `data/train/train.csv`, `data/train/X_train.npy`, `data/train/train_images.json`
-   `data/validation/validation.csv`, `data/validation/X_val.npy`, `data/validation/validation_images.json`
-   `data/test/test.csv`, `data/test/X_test.npy`, `data/test/test_images.json`
-   `docs/datasets/eda_report.json`, `docs/datasets/preprocessing_log.json`
-   `config/preprocessing_config.json`

## 3. Observații privind calitatea datelor

-   SMILES invalide sau lipsă eliminate: 6,811 observații
-   Valori lipsă semnificative în anumite coloane (ex. `Synonyms`, `Max Phase`) — tratate prin imputare mediană
-   Outlieri detectați și tratați prin IQR capping
-   Normalizare aplicată (Min-Max) pentru caracteristicile numerice și descriptorii moleculari

## 4. Utilizare imediată (scurt)

-   Date numerice pentru ML: `data/train/X_train.npy`, `data/validation/X_val.npy`, `data/test/X_test.npy`
-   Imagini 2D corelate (listă de căi): `data/*/*_images.json` (ex: `data/train/train_images.json`)
-   Date complete pentru vizualizare/analiză: `data/*/*.csv`

## 5. Următorii pași (detaliat și prioritizat)

1. Antrenare și validare modele (prioritate: ridicată)

-   Scop: antrenează modele GNN pe reprezentări grafice + experimente multimodale (imagine + descriptor numeric).
-   Componente: `ai_model/train.py` — verifică compatibilitate input cu `X_train.npy` și `train_images.json`.
-   Măsurători: precizie, recall, ROC-AUC, F1 (pe setul de validare și test).

2. Pipeline multimodal și augmentare imagini (prioritate: medie)

-   Include mecanism de fallback pentru probe fără imagine (`has_image` = 0).
-   Adaugă augmentări realiste (random rotation, crop, color jitter) și experimente comparative.

3. Instrumentare & evaluare robustă (prioritate: medie)

-   Adaugă unit/integration tests pentru `src/preprocessing`.
-   Instrumente de monitorizare pentru performanță modele și reproducibilitate (seed, config saved).

4. RAG / Document Retrieval (prioritate: medie)

-   Folosește `faiss_index/`, `embeddings.npy` și `chunks.json` pentru integrarea cu `backend/rag_helper.py`.
-   Evaluează acuratețea returnărilor (retrieval quality) și corelarea cu răspunsurile generate.

5. Mentenanță proiect și reproducibilitate (prioritate: scăzută)

-   Versionare dataset (semnături, hash), snapshot release pe GitHub.
-   Helm/Container: definire Dockerfile / compose pentru reproducere mediu de antrenare.

## 6. Stare Etapă (de completat de student)

-   [x] Structură repository configurată
-   [x] Dataset analizat (EDA realizată)
-   [x] Date preprocesate
-   [x] Seturi train/val/test generate
-   [x] Documentație actualizată în README + `data/README.md`

---

## 7. Cum reproduceți preprocesarea

1. Activează mediul Python:

```pwsh
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Rulează preprocesarea:

```pwsh
python src/preprocessing/data_preprocessing.py
```

## 8. Referință proiect

Pentru cod, date și context suplimentar, vedeți: https://github.com/DarkShadow1107/chemnet-vision

---

_Acest document oferă un rezumat orientat spre inginerie și pași concreți pentru continuarea dezvoltării._
