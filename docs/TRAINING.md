# Training

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/mini_ai_train data/train.txt mini_ai.ckpt
./build/mini_ai_test mini_ai.ckpt "Hola"
```

A checkpoint resumes both weights and Adam state. The trainer uses deterministic dataset sampling (seed 42) and the model uses deterministic initialization.
