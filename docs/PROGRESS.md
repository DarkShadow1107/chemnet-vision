# PROGRES SINTETIC — ChemNet Vision

**Data actualizare:** 2026-01-20

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Alexandru Gabriel  
**Data:** 2026-01-20

[![Repo](https://img.shields.io/badge/repo-DarkShadow1107/chemnet--vision-blue)](https://github.com/DarkShadow1107/chemnet-vision) [![Status](https://img.shields.io/badge/status-Stage%206%20complete-brightgreen)](https://github.com/DarkShadow1107/chemnet-vision) [![Dataset](https://img.shields.io/badge/dataset-optimized-orange)](https://github.com/DarkShadow1107/chemnet-vision/tree/main/data) [![Images](https://img.shields.io/badge/images-prepared-9cf)](https://github.com/DarkShadow1107/chemnet-vision/tree/main/data/2d_images)

## 1. Rezumat executiv

- Scop: Optimizarea sistemului de identificare moleculară prin antrenare extinsă și mecanisme hibride de inferență (Database + AI).
- Stare: Etapa 6 — Analiza performanței, optimizare și concluzii — finalizată.

Statistici cheie:

- Dataset inițial: 48,960 molecule (ChEMBL)
- Dataset final (SMILES valide): 42,149 molecule
- Împărțire: Train 29,503 | Validation 6,323 | Test 6,323
- Imagini 2D disponibile și corelate: 40,018 din 42,037 (acoperire ≈ 94.9%)

## 2. Fișiere cheie generate

- `models/nlp_model.pth` — model antrenat și optimizat pe setul complet.
- `results/optimization_experiments.csv` — log-ul celor 4 experimente de optimizare.
- `results/final_metrics.json` — performanțele finale (Accuracy ≈ 76%, Hybrid ≈ 85%).
- `README_Etape6_...md` — Documentația finală a proiectului.

## 3. Observații privind calitatea datelor (Etapa 6)

- **Eliminarea "Bluffing-ului":** S-a identificat o problemă de memorare a valorilor prototipice (ex. weight 859.14). Rezolvată prin extinderea setului de date și curățarea descriptorilor.
- **Acuratețe descriptori:** Utilizarea `MolWeight_RDKit` a oferit o precizie superioară față de datele brute din ChEMBL.

## 4. Utilizare imediată (scurt)

- Pornire servere: `start-servers.bat`
- Chat interactiv: Accesați `localhost:3000` și întrebați despre molecule (ex: "Water", "Caffeine").

## 5. Următorii pași (Finalizați în Etapa 6)

- [x] Antrenare pe tot setul (42,149 molecule).
- [x] Corectarea halucinațiilor prin Hybrid Inference.
- [x] Documentarea experimentelor de optimizare.
- [x] Implementarea formatului de răspuns structurat (Formula, Masă, Sinonime).
- [x] Pregătirea pentru prezentarea finală la examen.

## 6. Stare Etapă Finală

- [x] Model optimizat antrenat.
- [x] Backend hibrid funcțional.
- [x] Documentație performanță realizată.
- [x] State Machine actualizat.

---

## 7. Instrucțiuni Screenshots

Porniți aplicația și capturați următoarele în `docs/screenshots/`:

1. `inference_real.png` - Un chat despre Aspirină.
2. `inference_optimized.png` - Un chat despre Apă (confirmând greutatea 18.02).

---

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
