# Demonstratie End-to-End ChemNet-Vision

## Fluxul Demonstrat

### Scenariu 1: Describe Caffeine

| Pas | Actiune | Rezultat |
| --- | ------- | -------- |
| 1 | Utilizatorul scrie "Describe Caffeine" in chat | Input validat, trimis la backend Flask |
| 2 | PREPROCESS: Tokenizare query text | Textul este pregatit pentru modelul LSTM |
| 3 | INFERENCE: Forward pass prin LSTM | Modelul genereaza descriere textuala caracter-cu-caracter |
| 4 | CONFIDENCE_CHECK: Evaluare scor incredere | Confidence >= 0.85 -- predictie acceptata |
| 5 | GENERATE_VIZ: Creare vizualizari din SMILES `Cn1c(=O)c2c(ncn2C)n(C)c1=O` | Imagine 2D (RDKit) + structura 3D (SDF) generate |
| 6 | DISPLAY_RESULT | "CAFFEINE is a Small molecule. Formula: C8H10N4O2. Weight: 194.19 g/mol." + card molecula cu vizualizare 2D/3D |

### Scenariu 2: Describe Acetaminophen

| Pas | Actiune | Rezultat |
| --- | ------- | -------- |
| 1 | Utilizatorul scrie "Describe Acetaminophen" | Input validat, trimis la backend |
| 2 | PREPROCESS: Tokenizare query | Text pregatit pentru LSTM |
| 3 | INFERENCE: Forward pass LSTM | Generare descriere textuala cu informatii chimice |
| 4 | CONFIDENCE_CHECK: Scor ridicat | Predictie acceptata |
| 5 | GENERATE_VIZ: Vizualizare din SMILES `CC(=O)Nc1ccc(O)cc1` | Structura 2D si 3D generate |
| 6 | DISPLAY_RESULT | "ACETAMINOPHEN is a Small molecule. Formula: C8H9NO2. Weight: 151.16 g/mol." + vizualizare moleculara |

### Scenariu 3: Describe Naproxen

| Pas | Actiune | Rezultat |
| --- | ------- | -------- |
| 1 | Utilizatorul scrie "Describe Naproxen" | Input validat |
| 2 | PREPROCESS + INFERENCE | Modelul LSTM proceseaza si genereaza descrierea |
| 3 | CONFIDENCE_CHECK: Scor ridicat | Predictie acceptata |
| 4 | GENERATE_VIZ: Vizualizare moleculara | Structura 2D si 3D generate din SMILES |
| 5 | DISPLAY_RESULT | Informatii corecte despre Naproxen: formula, greutate moleculara, tip + card molecula cu vizualizare 2D/3D |

### Scenariu 4: Describe Ibuprofen

| Pas | Actiune | Rezultat |
| --- | ------- | -------- |
| 1 | Utilizatorul scrie "Describe Ibuprofen" | Input validat |
| 2 | PREPROCESS + INFERENCE | Modelul LSTM proceseaza si genereaza descrierea |
| 3 | CONFIDENCE_CHECK: Scor ridicat | Predictie acceptata |
| 4 | GENERATE_VIZ: Vizualizare moleculara | Structura 2D si 3D generate din SMILES |
| 5 | DISPLAY_RESULT | Informatii corecte despre Ibuprofen: formula, greutate moleculara, tip + card molecula cu vizualizare 2D/3D |

## Instructiuni de Reproducere

1. Porniti serverele: `start-servers.bat`
2. Accesati http://localhost:3000
3. In chat, scrieti: "Describe Caffeine"
4. Observati raspunsul structurat cu informatii chimice corecte
5. Incercati si alte molecule: "Describe Acetaminophen", "Describe Naproxen", "Describe Ibuprofen"

## Latenta Masurata

- Chat query (inferenta LSTM): ~35ms
- Generare vizualizare 2D/3D: ~50ms
- Total end-to-end: <100ms

## Data si Ora Demonstratiei

11.02.2026 23:17
