# results/

Acest folder conține artefactele generate de Etapa 5 (training + evaluare).

În acest moment nu rulez antrenarea (la cererea ta). După ce rulezi training/evaluare vei obține:

-   results/training_history.csv
-   results/hyperparameters.yaml (sau .json fallback)
-   results/test_metrics.json

Comenzi:

-   `python ai_model/train_model.py --epochs 20 --batch_size 8 --early_stopping --patience 5`
-   `python ai_model/evaluate.py --model models/trained_model.pth`
