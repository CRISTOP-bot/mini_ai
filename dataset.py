import random
from tokenizer import encode

class TextDataset:
    def __init__(self, path, context_size, seed=42):
        self.path = path
        self.context_size = context_size
        with open(path, 'r', encoding='utf-8') as f:
            self.tokens = encode(f.read())
        if len(self.tokens) < context_size + 1:
            raise ValueError(f'El dataset necesita al menos {context_size + 1} bytes.')
        self.rng = random.Random(seed)

    def sample(self, batch_size=1):
        xs, ys = [], []
        for _ in range(batch_size):
            start = self.rng.randrange(0, len(self.tokens) - self.context_size)
            xs.append(self.tokens[start:start+self.context_size])
            ys.append(self.tokens[start+1:start+self.context_size+1])
        return xs, ys

    def __len__(self):
        return len(self.tokens)
