# Screenshots

Acest director contine screenshot-uri demonstrative ale interfetei ChemNet-Vision.

## Screenshots necesare:

1. **ui_demo.png** - Interfata principala a aplicatiei (Etapa 4)
2. **inference_real.png** - Inferenta model antrenat (Etapa 5)
3. **inference_optimized.png** - Inferenta model optimizat cu card molecula (Etapa 6)

## Cum sa adaugi screenshots:

1. Porneste aplicatia:

    ```bash
    # Terminal 1
    python backend/app.py

    # Terminal 2
    npm run dev
    ```

2. Deschide http://localhost:3000

3. Fa screenshot-uri si salveaza-le aici

## Exemple de functionalitati de capturat:

- Chat interface cu mesaje
- Card molecula cu informatii structurate (nume, formula, greutate)
- Vizualizator 2D (RDKit) si 3D (Py3Dmol) molecule
- Interogare molecule: "Describe Caffeine", "Tell me about Aspirin"
