import argparse
import math
import random
from config import VOCAB_SIZE, CONTEXT_SIZE
from tokenizer import encode, decode
from model import TinyTransformer
from save import load_model

def choose_token(row, temperature=0.0, top_k=0):
    """Greedy si temperature=0; sampling estable opcional para más variedad."""
    if temperature <= 0:
        return max(range(len(row)), key=lambda i: row[i])
    values = [(i, x / temperature) for i, x in enumerate(row)]
    if top_k > 0:
        values = sorted(values, key=lambda x: x[1], reverse=True)[:top_k]
    mx = max(x for _, x in values)
    weights = [math.exp(x - mx) for _, x in values]
    return random.choices([i for i, _ in values], weights=weights, k=1)[0]

def main():
    parser = argparse.ArgumentParser(description='Generador autoregresivo de Mini IA')
    parser.add_argument('prompt', nargs='*', help='texto inicial')
    parser.add_argument('--tokens', type=int, default=100, help='bytes nuevos (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='0=greedy; valores como 0.8 activan sampling')
    parser.add_argument('--top-k', type=int, default=0,
                        help='limitar sampling a los k tokens más probables')
    args = parser.parse_args()
    prompt = ' '.join(args.prompt) if args.prompt else input('Prompt: ')
    if not prompt.strip():
        prompt = 'hola'
    if args.tokens < 0 or args.tokens > 10000:
        parser.error('--tokens debe estar entre 0 y 10000')
    if args.temperature < 0:
        parser.error('--temperature no puede ser negativa')
    if args.top_k < 0 or args.top_k > VOCAB_SIZE:
        parser.error(f'--top-k debe estar entre 0 y {VOCAB_SIZE}')

    model = TinyTransformer()
    load_model(model, 'models/model.bin')
    tokens = encode(prompt)
    for _ in range(args.tokens):
        ctx = tokens[-CONTEXT_SIZE:]
        logits = model.forward(ctx)
        last_start = (len(ctx) - 1) * VOCAB_SIZE
        row = logits.data[last_start:last_start + VOCAB_SIZE]
        tokens.append(choose_token(row, args.temperature, args.top_k))
    print('IA:', decode(tokens))

if __name__ == '__main__':
    main()
