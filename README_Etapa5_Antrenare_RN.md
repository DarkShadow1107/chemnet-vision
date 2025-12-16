# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** [URL complet]  
**Data predării:** [Data]

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:

-   State Machine definit și justificat
-   Cele 3 module funcționale (Data Logging, RN, UI)
-   Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

-   [ ] **State Machine** definit și documentat în `docs/state_machine.*`
-   [ ] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
-   [ ] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
-   [ ] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.pth`)
-   [ ] **Modul 3 (UI/Web Service)** funcțional cu model dummy
-   [ ] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

Notă (clarificare importantă):

-   **Arhitectura** RN este definită în `ai_model/model.py`.
-   Fișierul `models/untrained_model.pth` este un **checkpoint cu weights random** (aceeași arhitectură, dar fără antrenare). În repo-ul acesta el există deja și se poate regenera rulând `ai_model/train_model.py` (scriptul salvează un checkpoint neantrenat dacă lipsește).

## Pregătire Date pentru Antrenare

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

**TREBUIE să refaceți preprocesarea pe dataset-ul COMBINAT:**

Exemplu:

```bash
# (Re)generare processed + split train/val/test
python src/preprocessing/data_preprocessing.py

# Verificare finală:
# data/train/ → trebuie să conțină date vechi + noi
# data/validation/ → trebuie să conțină date vechi + noi
# data/test/ → trebuie să conțină date vechi + noi
```

** ATENȚIE - Folosiți ACEIAȘI parametri de preprocesare:**

-   Aceiași parametri de preprocesare (config): `config/preprocessing_config.json`
-   Aceiași proporții split: 70% train / 15% validation / 15% test
-   Același `random_state=42` pentru reproducibilitate

**Verificare rapidă:**

```python
import pandas as pd
train = pd.read_csv('data/train/train.csv')
print(f"Train samples: {len(train)}")  # Trebuie să includă date noi
```

---

## Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

Completați **TOATE** punctele următoare:

1. **Antrenare model** definit în Etapa 4 pe setul final de date (≥40% originale)
2. **Minimum 10 epoci**, batch size 8–32
3. **Împărțire stratificată** train/validation/test: 70% / 15% / 15%
4. **Tabel justificare hiperparametri** (vezi secțiunea de mai jos - OBLIGATORIU)
5. **Metrici calculate pe test set:**
    - **Acuratețe ≥ 65%**
    - **F1-score (macro) ≥ 0.60**
6. **Salvare model antrenat** în `models/trained_model.pth` (PyTorch)
7. **Integrare în UI din Etapa 4:**
    - UI trebuie să încarce modelul ANTRENAT (nu dummy)
    - Inferență REALĂ demonstrată
    - Screenshot în `docs/screenshots/inference_real.png`

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

Completați tabelul cu hiperparametrii folosiți și **justificați fiecare alegere**:

| **Hiperparametru**   | **Valoare Aleasă**                  | **Justificare**                                                         |
| -------------------- | ----------------------------------- | ----------------------------------------------------------------------- |
| Learning rate        | Ex: 0.001                           | Valoare standard pentru Adam optimizer, asigură convergență stabilă     |
| Batch size           | Ex: 32                              | Compromis memorie/stabilitate pentru N=[numărul vostru] samples         |
| Number of epochs     | Ex: 50                              | Cu early stopping după 10 epoci fără îmbunătățire                       |
| Optimizer            | Ex: Adam                            | Adaptive learning rate, potrivit pentru RN cu [numărul vostru] straturi |
| Loss function        | Ex: Categorical Crossentropy        | Clasificare multi-class cu K=[numărul vostru] clase                     |
| Activation functions | Ex: ReLU (hidden), Softmax (output) | ReLU pentru non-linearitate, Softmax pentru probabilități clase         |

**Justificare detaliată batch size (exemplu):**

```
Am ales batch_size=32 pentru că avem N=15,000 samples → 15,000/32 ≈ 469 iterații/epocă.
Aceasta oferă un echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Memorie GPU (batch prea mare → out of memory)
- Timp antrenare (batch 32 asigură convergență în ~50 epoci pentru problema noastră)
```

**Resurse învățare rapidă:**

-   Împărțire date: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html (video 3 min: https://youtu.be/1NjLMWSGosI?si=KL8Qv2SJ1d_mFZfr)
-   Antrenare simplă Keras: https://keras.io/examples/vision/mnist_convnet/ (secțiunea „Training”)
-   Antrenare simplă PyTorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-an-image-classifier (video 2 min: https://youtu.be/ORMx45xqWkA?si=FXyQEhh0DU8VnuVJ)
-   F1-score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html (video 4 min: https://youtu.be/ZQlEcyNV6wc?si=VMCl8aGfhCfp5Egi)

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutive
2. **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
3. **Augmentări relevante domeniu:**
    - Vibrații motor: zgomot gaussian calibrat, jitter temporal
    - Imagini industriale: slight perspective, lighting variation (nu rotații simple!)
    - Serii temporale: time warping, magnitude warping
4. **Grafic loss și val_loss** în funcție de epoci salvat în `docs/loss_curve.png`
5. **Analiză erori context industrial** (vezi secțiunea dedicată mai jos - OBLIGATORIU Nivel 2)

**Indicatori țintă Nivel 2:**

-   **Acuratețe ≥ 75%**
-   **F1-score (macro) ≥ 0.70**

**Resurse învățare (aplicații industriale):**

-   Albumentations: https://albumentations.ai/docs/examples/
-   Early Stopping + ReduceLROnPlateau în Keras: https://keras.io/api/callbacks/
-   Scheduler în PyTorch: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

---

### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate**                               | **Livrabil**                                            |
| -------------------------------------------- | ------------------------------------------------------- |
| Comparare 2+ arhitecturi diferite            | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență       | Fișier `models/final_model.onnx` + demonstrație <50ms   |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README         |

**Resurse bonus:**

-   Export ONNX din PyTorch: [PyTorch ONNX Tutorial](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
-   TensorFlow Lite converter: [TFLite Conversion Guide](https://www.tensorflow.org/lite/convert)
-   Confusion Matrix analiză: [Scikit-learn Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

| **Stare din Etapa 4** | **Implementare în ChemNet Vision**                                                                        |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| `ACQUIRE_DATA`        | Încărcare dataset din `data/train/`, `data/validation/`, `data/test/`                                     |
| `PREPROCESS`          | Preprocesare controlată de `src/preprocessing/data_preprocessing.py` + `config/preprocessing_config.json` |
| `RN_INFERENCE`        | Inferență prin `ai_model/inference.py` (folosit de `backend/app.py`)                                      |
| `THRESHOLD_CHECK`     | Selectarea modului AI/Fallback/Auto și validări de output în backend                                      |
| `ALERT`               | Mesaje/răspuns în UI (Next.js) pe baza rezultatului din backend                                           |

Verificați că backend-ul încarcă modelul antrenat:

-   Preferă `models/trained_model.pth`
-   Fallback: `saved_models/checkpoint_best.pth`

Referințe în cod: `backend/app.py`, `ai_model/inference.py`.

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Exemplu robotică (predicție traiectorii):**

```
Confusion Matrix arată că modelul confundă 'viraj stânga' cu 'viraj dreapta' în 18% din cazuri.
Cauză posibilă: Features-urile IMU (gyro_z) sunt simetrice pentru viraje în direcții opuse.
```

**Completați pentru proiectul vostru:**

```
[Descrieți confuziile principale între clase și cauzele posibile]
```

### 2. Ce caracteristici ale datelor cauzează erori?

**Exemplu vibrații motor:**

```
Modelul eșuează când zgomotul de fond depășește 40% din amplitudinea semnalului util.
În mediul industrial, acest nivel de zgomot apare când mai multe motoare funcționează simultan.
```

**Completați pentru proiectul vostru:**

```
[Identificați condițiile în care modelul are performanță slabă]
```

### 3. Ce implicații are pentru aplicația industrială?

**Exemplu detectare defecte sudură:**

```
FALSE NEGATIVES (defect nedetectat): CRITIC → risc rupere sudură în exploatare
FALSE POSITIVES (alarmă falsă): ACCEPTABIL → piesa este re-inspectată manual

Prioritate: Minimizare false negatives chiar dacă cresc false positives.
Soluție: Ajustare threshold clasificare de la 0.5 → 0.3 pentru clasa 'defect'.
```

**Completați pentru proiectul vostru:**

```
[Analizați impactul erorilor în contextul aplicației voastre și prioritizați]
```

### 4. Ce măsuri corective propuneți?

**Exemplu clasificare imagini piese:**

```
Măsuri corective:
1. Colectare 500+ imagini adiționale pentru clasa minoritară 'zgârietură ușoară'
2. Implementare filtrare Gaussian blur pentru reducere zgomot cameră industrială
3. Augmentare perspective pentru simulare unghiuri camera variabile (±15°)
4. Re-antrenare cu class weights: [1.0, 2.5, 1.2] pentru echilibrare
```

**Completați pentru proiectul vostru:**

```
[Propuneți minimum 3 măsuri concrete pentru îmbunătățire]
```

---

## Structura Repository-ului la Finalul Etapei 5

Structura din acest repository (ChemNet Vision) folosește **PyTorch** și păstrează scripturile de RN în folderul `ai_model/`.

```
chemnet-vision/
├── README.md
├── README_Etapa4_Arhitectura_SIA_03.12.2025.md
├── README_Etapa5_Antrenare_RN.md               # ← ACEST FIȘIER
│
├── ai_model/
│   ├── model.py                                # Arhitectură (CNN+MLP+GNN+LSTM)
│   ├── train_model.py                           # Script antrenare (Etapa 5)
│   ├── evaluate.py                              # Script evaluare (Etapa 5)
│   └── inference.py                             # Inference (folosit de backend/UI)
│
├── backend/
│   ├── app.py                                   # Flask API; încarcă modelul antrenat
│   └── rag_helper.py
│
├── src/                                         # UI Next.js
│   └── app/
│       ├── page.tsx
│       └── api/
│           └── conversations/route.ts
│
├── docs/
│   ├── state_machine.md                         # Etapa 4 (documentat)
│   ├── loss_curve.png                            # Nivel 2 (generat după antrenare)
│   ├── learning_curves.png                        # Curbe învățare (loss + accuracy)
│   ├── confusion_matrix.png                      # Nivel 3 (opțional)
│   ├── test_class_distribution.png                # Distribuție clase pe test (token-uri)
│   └── screenshots/
│       └── inference_real.png                    # Nivel 1 (după demonstrație)
│
├── data/
│   ├── raw/
│   ├── generated/                                # Contribuția voastră 40% (derivate/artefacte)
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── models/
│   ├── untrained_model.pth                       # weights random (Etapa 4)
│   ├── trained_model.pth                         # checkpoint best (Etapa 5)
│   └── vocab.json                                # vocab folosit la decodare
│
├── results/
│   ├── training_history.csv                      # istoric pe epoci
│   ├── test_metrics.json                         # metrici pe test set
│   └── hyperparameters.yaml                      # hiperparametri folosiți
│
├── requirements.txt
└── start-servers.bat
```

**Diferențe față de Etapa 4:**

-   Actualizat antrenarea/evaluarea în `ai_model/` (nu există `src/neural_network/` în acest repo)
-   Adăugat `models/trained_model.pth` - OBLIGATORIU
-   Adăugat `results/` cu history + metrici + hiperparametri
-   Generare artefacte în `docs/` (`loss_curve.png`, opțional `confusion_matrix.png`)
-   Backend-ul și inferența preferă `models/trained_model.pth` când există

---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

### 1. Setup mediu (dacă nu ați făcut deja)

```bash
pip install -r requirements.txt
```

### 2. Pregătire date (DACĂ ați adăugat date noi în Etapa 4)

```bash
# (Re)generare processed + split train/val/test
python src/preprocessing/data_preprocessing.py
```

### 3. Antrenare model

```bash
python ai_model/train_model.py --epochs 50 --batch_size 32 --early_stopping --patience 5

# Output așteptat:
# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823
# ...
# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956
# Early stopping triggered at epoch 23
# ✓ Model saved to models/trained_model.pth
```

### 4. Evaluare pe test set

```bash
python ai_model/evaluate.py --model models/trained_model.pth

# Output așteptat:
# Token Accuracy: 0.78
# Token F1-score (macro): 0.74
# ✓ Metrics saved to results/test_metrics.json
# ✓ Confusion matrix saved to docs/confusion_matrix.png
```

Notă metrici (important pentru rubrică): acest proiect generează secvențe (SMILES), deci **Accuracy/F1 sunt calculate token-level (teacher-forced)**, excluzând tokenul `<pad>`.

### 5. Lansare UI cu model antrenat

```bash
# Porniți backend + UI (Next.js)

# opțiunea 1 (recomandat):
start-servers.bat

# opțiunea 2 (manual, în 2 terminale):
# Terminal A:
#   python backend/app.py
# Terminal B:
#   npm install
#   npm run dev
```

**Testare în UI:**

1. Introduceți date de test (manual sau upload fișier)
2. Verificați că predicția este DIFERITĂ de Etapa 4 (când era random)
3. Verificați că confidence scores au sens (ex: 85% pentru clasa corectă)
4. Faceți screenshot → salvați în `docs/screenshots/inference_real.png`

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)

-   [ ] State Machine există și e documentat în `docs/state_machine.*`
-   [ ] Contribuție ≥40% date originale verificabilă în `data/generated/`
-   [ ] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date

-   [ ] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
-   [ ] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
-   [ ] Config de preprocesare folosit consistent (`config/preprocessing_config.json`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)

-   [ ] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
-   [ ] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
-   [ ] Tabel hiperparametri + justificări completat în acest README
-   [ ] Metrici calculate pe test set (token-level): **Accuracy ≥65%**, **F1 ≥0.60**
-   [ ] Model salvat în `models/trained_model.pth`
-   [ ] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)

-   [ ] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
-   [ ] UI face inferență REALĂ cu predicții corecte
-   [ ] Screenshot inferență reală în `docs/screenshots/inference_real.png`
-   [ ] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)

-   [ ] Early stopping implementat și documentat în cod
-   [ ] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
-   [ ] Augmentări relevante domeniu aplicate (NU rotații simple!)
-   [ ] Grafic loss/val_loss salvat în `docs/loss_curve.png`
-   [ ] Analiză erori în context industrial completată (4 întrebări răspunse)
-   [ ] Metrici Nivel 2 (token-level): **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)

-   [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
-   [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
-   [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice

-   [ ] `requirements.txt` actualizat cu toate bibliotecile noi
-   [ ] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
-   [ ] Cod nou comentat în limba română sau engleză (minimum 15%)
-   [ ] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
-   [ ] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)

-   [ ] Fluxul de inferență respectă stările din State Machine
-   [ ] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
-   [ ] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare

-   [ ] `README_Etapa5_Antrenare_RN.md` completat cu TOATE secțiunile
-   [ ] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
-   [ ] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
-   [ ] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
-   [ ] Push: `git push origin main --tags`
-   [ ] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`README_Etapa5_Antrenare_RN.md`** (acest fișier) cu:

    - Tabel hiperparametri + justificări (complet)
    - Metrici test set raportate (token-level accuracy, token-level F1)
    - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.pth`** - model antrenat funcțional (PyTorch)

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:

```json
{
	"token_accuracy": 0.7823,
	"token_f1_macro": 0.7456,
	"token_precision_macro": 0.7612,
	"token_recall_macro": 0.7321,
	"valid_smiles_rate": 0.58,
	"notes": "Token-level metrics (teacher-forced) for SMILES generation; <pad> excluded."
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

6b. **(Recomandat)** `docs/learning_curves.png` - learning curves (loss + accuracy)

6c. **(Recomandat)** `docs/test_class_distribution.png` - distribuție clase pe test (token-uri)

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**

1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**
