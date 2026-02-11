# Etapa 5 -- Antrenarea Modelului

**Proiect:** ChemNet-Vision
**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti -- FIIR
**Student:** Alexandru Gabriel
**Data:** 2026-02-11

---

## 1. Prezentare generala

Aceasta etapa documenteaza procesul de antrenare a modelelor ChemNet-Vision: modelul multimodal principal si modelul NLP auxiliar. Antrenarea s-a realizat pe setul de date preprocesat de 42,149 molecule, generand aproximativ 120,000 de perechi de antrenare.

---

## 2. Dataset de antrenare

| Parametru                     | Valoare      |
| ----------------------------- | ------------ |
| Molecule valide               | 42,149       |
| Perechi de antrenare generate | ~120,000     |
| Split Train                   | 29,503 (70%) |
| Split Validation              | 6,323 (15%)  |
| Split Test                    | 6,323 (15%)  |

Perechile de antrenare includ combinatii de imagini 2D, descriptori moleculari si secvente SMILES tinta.

---

## 3. Configuratie antrenare -- Model principal (multimodal)

### 3.1. Hiperparametri

| Parametru     | Valoare           |
| ------------- | ----------------- |
| Learning Rate | 0.001             |
| Batch Size    | 8                 |
| Epochs        | 20 (initial)      |
| Optimizer     | Adam              |
| Scheduler     | ReduceLROnPlateau |
| Loss Function | CrossEntropyLoss  |
| Device        | CUDA (GPU) / CPU  |

### 3.2. Scheduler ReduceLROnPlateau

- **Mode:** min (monitorizare validation loss)
- **Factor:** 0.5 (reduce LR la jumatate)
- **Patience:** 3 epoci fara imbunatatire
- **Min LR:** 1e-6

### 3.3. Observatii

- Batch size de 8 a fost ales datorita dimensiunii mari a imaginilor (224x224) si limitarilor de memorie GPU.
- Antrenarea initiala pe 20 de epoci a servit ca baseline; extensia ulterioara s-a realizat in Etapa 6.

---

## 4. Configuratie antrenare -- Model NLP

### 4.1. Hiperparametri

| Parametru     | Valoare          |
| ------------- | ---------------- |
| Learning Rate | 0.002            |
| Batch Size    | 256              |
| Epochs        | 50               |
| Optimizer     | Adam             |
| Scheduler     | StepLR           |
| Loss Function | CrossEntropyLoss |

### 4.2. Scheduler StepLR

- **Step Size:** 10 epoci
- **Gamma:** 0.5

### 4.3. Arhitectura NLP (recapitulare)

```
Input Token -> Embedding(128) -> LSTM(hidden=256) -> Linear(vocab_size) -> Output Token
```

Modelul NLP genereaza informatii structurate (formula chimica, greutate moleculara, sinonime) pe baza numelui moleculei.

---

## 5. Metrici baseline

Dupa antrenarea initiala, metricile pe setul de test au fost:

| Metrica      | Valoare |
| ------------ | ------- |
| **Accuracy** | 72%     |
| **F1 Score** | 0.68    |

Aceste valori constituie baseline-ul pentru experimentele de optimizare din Etapa 6.

---

## 6. Istoricul antrenarii

Evolutia loss-ului si a metricilor pe parcursul epocilor este stocata in:

```
results/training_history.csv
```

Fisierul contine urmatoarele coloane:

- `epoch` -- numarul epocii
- `train_loss` -- loss pe setul de antrenare
- `val_loss` -- loss pe setul de validare
- `val_accuracy` -- acuratatea pe setul de validare
- `learning_rate` -- rata de invatare curenta

---

## 7. Model salvat

### 7.1. Model NLP antrenat

| Parametru    | Valoare                |
| ------------ | ---------------------- |
| Fisier       | `models/nlp_model.pth` |
| Dimensiune   | 1.73 MB                |
| Epoca finala | 50                     |

### 7.2. Continutul checkpoint-ului

Fisierul `.pth` contine:

- `model_state_dict` -- ponderile modelului
- `optimizer_state_dict` -- starea optimizatorului
- `vocab` -- vocabularul de caractere utilizat
- `config` -- configuratia modelului (embedding_dim, hidden_dim, etc.)

---

## 8. Integrare cu backend-ul Flask

Modelul antrenat a fost integrat in backend-ul Flask (`backend/app.py`) cu suport pentru 3 moduri de operare:

### 8.1. Moduri de inferenta

| Mod          | Descriere                                                  |
| ------------ | ---------------------------------------------------------- |
| **ai**       | Inferenta exclusiv prin reteaua neurala antrenata          |
| **fallback** | Raspuns exclusiv din baza de date locala (fara model)      |
| **auto**     | Inferenta AI cu fallback automat la DB daca SMILES invalid |

### 8.2. Fluxul de inferenta (mod `auto`)

1. Se primeste query-ul utilizatorului.
2. Se ruleaza modelul NLP pe query.
3. Se valideaza raspunsul generat.
4. Daca raspunsul este valid -> se returneaza.
5. Daca raspunsul este invalid -> se cauta in baza de date locala.
6. Se genereaza vizualizarea 2D/3D.
7. Se returneaza raspunsul structurat (JSON).

### 8.3. Endpoint API

```
POST /api/chat
Content-Type: application/json

{
    "message": "Tell me about Caffeine",
    "mode": "auto"
}
```

Raspunsul contine: formula chimica, greutate moleculara, SMILES canonic, sinonime si URL imagine 2D.

---

## 9. Diagrama procesului de antrenare

```
Dataset (42,149 molecule)
    |
    v
Generare perechi (~120k)
    |
    +--> Train (70%) -----> Antrenare model -----> Checkpoint
    |                           |
    +--> Validation (15%) ---> Evaluare periodica ---> ReduceLROnPlateau
    |
    +--> Test (15%) ---------> Evaluare finala -----> Metrici baseline
```

---

## 10. Concluzii Etapa 5

- Modelul NLP a fost antrenat cu succes pe 50 de epoci, atingand **Accuracy 72%** si **F1 Score 0.68** ca baseline.
- **ReduceLROnPlateau** si **StepLR** au asigurat convergenta stabila a procesului de antrenare.
- Modelul salvat (`models/nlp_model.pth`, 1.73 MB) este suficient de compact pentru deployment.
- Integrarea cu Flask permite inferenta in timp real prin cele 3 moduri de operare.
- Istoricul antrenarii este disponibil in `results/training_history.csv` pentru analiza ulterioara.
- Metricile baseline vor fi imbunatatite in Etapa 6 prin experimente de optimizare.

---

_Document generat pentru Etapa 5 -- Antrenarea Modelului, disciplina Retele Neuronale, POLITEHNICA Bucuresti._
