# Etapa 3 -- Analiza Datelor

**Proiect:** ChemNet-Vision
**Disciplina:** Retele Neuronale
**Institutie:** POLITEHNICA Bucuresti -- FIIR
**Student:** Alexandru Gabriel
**Data:** 2026-02-11

---

## 1. Sursa datelor

Setul de date provine din **ChEMBL Database**, o baza de date publica de compusi bioactivi, gestionata de European Bioinformatics Institute (EMBL-EBI). Datele au fost extrase si prelucrate local in cadrul proiectului.

| Parametru                    | Valoare                    |
| ---------------------------- | -------------------------- |
| Sursa                        | ChEMBL Database            |
| Numar initial de molecule    | 48,960                     |
| Dupa filtrare/validare RDKit | 42,149                     |
| Format initial               | CSV (SMILES + descriptori) |

---

## 2. Filtrare si validare

Procesul de filtrare a fost realizat cu ajutorul bibliotecii **RDKit** si a constat in:

1. **Validare SMILES** -- eliminarea moleculelor cu notatii SMILES invalide sau neparsabile.
2. **Eliminare duplicate** -- pe baza SMILES canonic.
3. **Filtrare greutate moleculara** -- eliminarea moleculelor cu valori aberante extreme.
4. **Verificare consistenta** -- corelarea numelui cu structura chimica.

Rezultat: din 48,960 molecule initiale, **42,149 molecule** au trecut validarea completa.

---

## 3. Features (Caracteristici)

Setul de date final contine **23 de features**, provenind din doua surse:

### 3.1. Features din ChEMBL (13)

Acestea sunt descriptori extrasi direct din baza de date ChEMBL, incluzand:

- Molecule ChEMBL ID
- Molecule Name
- Molecular Formula
- Molecular Weight (ChEMBL)
- SMILES canonic
- ALogP
- HBA (Hydrogen Bond Acceptors)
- HBD (Hydrogen Bond Donors)
- PSA (Polar Surface Area)
- RO5 Violations
- Aromatic Rings
- Heavy Atoms
- Molecule Type

### 3.2. Features calculate cu RDKit (10)

Descriptori moleculari calculati local prin biblioteca RDKit:

- MolWeight_RDKit (greutate moleculara recalculata)
- LogP_RDKit
- NumHDonors
- NumHAcceptors
- TPSA (Topological Polar Surface Area)
- NumRotatableBonds
- NumAromaticRings
- FractionCSP3
- NumHeavyAtoms
- MolLogP

---

## 4. Impartirea setului de date

Setul de date a fost impartit in trei subseturi, utilizand stratificare pentru a mentine distributia claselor:

| Subset         | Numar molecule | Procent |
| -------------- | -------------- | ------- |
| **Train**      | 29,503         | 70%     |
| **Validation** | 6,323          | 15%     |
| **Test**       | 6,323          | 15%     |

Impartirea se realizeaza reproducibil prin seed fix (`random_state=42`).

---

## 5. Preprocesare

### 5.1. Normalizare

- **Min-Max Normalization** aplicata pe toate features numerice, scaland valorile in intervalul [0, 1].

### 5.2. Codificare

- **One-Hot Encoding** pentru variabilele categorice (ex: Molecule Type).

### 5.3. Imputare valori lipsa

- **Imputare cu mediana** pentru features numerice cu valori lipsa.
- Mediana a fost aleasa in detrimentul mediei pentru robustete la outlieri.

### 5.4. Tratarea outlierilor

- **IQR Outlier Capping** (Interquartile Range): valorile sub Q1 - 1.5*IQR sau peste Q3 + 1.5*IQR sunt limitate (capped) la aceste praguri.
- Aceasta metoda pastreaza toate observatiile dar limiteaza influenta extremelor.

### 5.5. Validare SMILES

- Fiecare SMILES este parsat cu `Chem.MolFromSmiles()` din RDKit.
- Moleculele cu SMILES invalid sunt eliminate din setul de date.
- SMILES-urile sunt canonizate pentru consistenta.

---

## 6. Analiza exploratorie (EDA)

### 6.1. Distributia greutatii moleculare

| Statistica | Valoare      |
| ---------- | ------------ |
| Minim      | 4.0 g/mol    |
| Maxim      | 859.14 g/mol |
| Mediana    | ~350 g/mol   |

### 6.2. Observatii cheie

- **Bias catre molecule grele**: in datele brute, distributia greutatii moleculare este deplasata catre valori mari (>400 g/mol), ceea ce reflecta compozitia bazei ChEMBL (compusi farmaceutici).
- **Distributie asimetrica**: majoritatea features prezinta distributii asimetrice pozitive (right-skewed), justificand normalizarea Min-Max si capping-ul IQR.
- **Corelatii puternice**: MolWeight si NumHeavyAtoms prezinta corelatie ridicata (>0.9), la fel ALogP si MolLogP_RDKit.
- **Valori lipsa**: sub 2% din observatii prezentau valori lipsa, tratate prin imputare cu mediana.

---

## 7. Fisiere generate

| Fisier                                   | Descriere                                   |
| ---------------------------------------- | ------------------------------------------- |
| `data/processed/molecules_processed.csv` | Setul de date final, preprocesat si validat |
| `config/preprocessing_config.json`       | Configuratia pipeline-ului de preprocesare  |

---

## 8. Script de preprocesare

Intregul pipeline de preprocesare este implementat in:

```
src/preprocessing/data_preprocessing.py
```

Scriptul realizeaza toate etapele descrise mai sus (validare, filtrare, normalizare, codificare, imputare, capping, split) si poate fi rulat independent pentru a reproduce rezultatele.

### Utilizare:

```bash
python src/preprocessing/data_preprocessing.py
```

---

## 9. Concluzii Etapa 3

- Setul de date final de **42,149 molecule** este curat, validat chimic si pregatit pentru antrenarea retelei neuronale.
- Cele **23 de features** ofera o descriere comprehensiva a fiecarei molecule, combinand date din ChEMBL cu descriptori calculati local prin RDKit.
- Preprocesarea robusta (normalizare, imputare, capping) asigura calitatea datelor de intrare pentru model.
- Impartirea 70/15/15 permite atat antrenare pe un set suficient de mare, cat si evaluare fidela pe seturi de validare si test separate.

---

_Document generat pentru Etapa 3 -- Analiza Datelor, disciplina Retele Neuronale, POLITEHNICA Bucuresti._
