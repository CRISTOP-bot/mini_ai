# Mini IA propia para Termux

Red neuronal autoregresiva educativa, entrenada desde cero y sin modelos externos. Solo usa Python estándar. El tokenizer convierte UTF-8 a bytes (256 tokens). El modelo tiene embeddings, posiciones, atención causal, feed-forward `tanh`, salida lineal, softmax implícito en la entropía cruzada y backpropagation con Adam.

## Termux
```sh
pkg update
pkg install python
termux-setup-storage   # opcional, solo si tus textos están en el almacenamiento compartido
cd mini_ai
python test.py
python train.py data/train.txt
python generate.py
python generate.py "hola"
# Más variedad: temperatura y top-k
python generate.py --temperature 0.8 --top-k 20 --tokens 120 "hola"
```
No requiere gcc, cmake, NumPy, CUDA, TensorFlow ni PyTorch. `STEPS=100 python train.py data/train.txt` sirve para pruebas rápidas. En teléfonos modestos empieza con 100–1000 pasos; Python puro es deliberadamente lento.

## Archivos y uso
- `tokenizer.py`: UTF-8 ↔ bytes.
- `dataset.py`: lee el texto y toma ventanas aleatorias de 32 bytes (no crea todos los batches).
- `tensor.py`: Tensor plano y grafo de autodiferenciación; incluye matmul, activaciones, softmax y cross-entropy.
- `model.py`: transformer diminuto causal.
- `optimizer.py`: Adam simplificado.
- `save.py`: checkpoint local con pickle.
- `train.py`: entrenamiento: `python train.py [archivo]`.
- `generate.py`: generación autoregresiva usando el checkpoint.

El dataset debe tener al menos 33 bytes. Sustitúyelo con `cp mi_texto.txt data/train.txt` (UTF-8). No hace falta reentrenar un tokenizer.

## Aumentar tamaño
Edita `config.py`: `CONTEXT_SIZE`, `EMBEDDING_DIM` y `HIDDEN_DIM` (esta primera implementación fija una sola capa; `NUM_LAYERS` queda reservado para una siguiente extensión). Aumentar dimensiones consume tiempo y RAM cuadráticamente en la atención; borra `models/model.bin` y entrena desde cero después de cambiar la configuración.

## Qué aprende
Cada embedding y cada matriz empieza con números aleatorios. El error compara la distribución de salida con el byte correcto; backpropagation calcula cómo cambiar cada número y Adam los actualiza. Después de muchos ejemplos, esos pesos codifican asociaciones estadísticas de bytes y posiciones del dataset. No es memoria literal garantizada ni una IA general: con un dataset pequeño aprenderá sobre todo patrones y frases frecuentes.
