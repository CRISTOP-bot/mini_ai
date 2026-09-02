# Entrenamiento autónomo

## Entrenamiento local

Desde la raíz del proyecto:

```sh
python test.py
python train.py --auto --max-steps 5000 --patience 10 --min-delta 0.001 --eval-interval 100
```

El entrenador divide las ventanas disponibles con una semilla fija. Las ventanas de validación no actualizan los pesos. La métrica principal es `val_loss`.

## Early stopping

En cada evaluación, una mejora solo cuenta cuando:

```text
val_loss < mejor_val_loss - min_delta
```

Una mejora guarda `best_model.bin` y reinicia el contador. Si se acumulan `patience` evaluaciones sin mejora, el entrenamiento termina. `max_steps` es siempre el límite absoluto.

## Checkpoints

```text
checkpoints/
├── best_model.bin
├── latest_model.bin
└── training_metrics.jsonl
```

`latest_model.bin` permite continuar con el estado del optimizador:

```sh
python train.py --auto --resume latest_model.bin --checkpoint-dir checkpoints --max-steps 5000
```

El archivo de métricas usa JSONL y registra `step`, `train_loss`, `val_loss`, `best_val_loss` y `learning_rate`.

## GitHub Actions

El workflow `.github/workflows/train.yml` se ejecuta manualmente desde Actions > Entrenamiento autónomo > Run workflow. Sus entradas son `max_steps` y `patience`. El job ejecuta las pruebas, investiga Wikipedia, entrena y publica un artifact con checkpoints, métricas y datos de investigación.

El workflow usa CPU y tiene un límite de tiempo. No instala PyTorch, TensorFlow, CUDA ni otras dependencias grandes.
