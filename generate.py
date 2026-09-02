import argparse
import codecs
import math
import random
from config import VOCAB_SIZE, CONTEXT_SIZE
from tokenizer import encode, decode
from model import TinyTransformer
from save import load_model

def valid_text_byte(history, candidate):
    """Evita bytes UTF-8 inválidos y controles raros durante la generación."""
    if candidate < 32 and candidate not in (9, 10, 13):
        return False
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    try:
        decoder.decode(bytes(history + [candidate]), final=False)
        return True
    except UnicodeDecodeError:
        return False

def choose_token(row, history, text_bytes, temperature=0.0, top_k=0):
    # Mantiene el vocabulario byte-level, pero evita inventar bytes que nunca
    # aparecieron en el corpus (por ejemplo, alfabetos ajenos al español).
    allowed = [i for i in range(len(row))
               if i in text_bytes and valid_text_byte(history, i)]
    if not allowed:
        allowed = list(range(len(row)))
    if temperature <= 0:
        return max(allowed, key=lambda i: row[i])
    values = [(i, row[i] / temperature) for i in allowed]
    if top_k > 0:
        values = sorted(values, key=lambda x: x[1], reverse=True)[:top_k]
    mx = max(x for _, x in values)
    weights = [math.exp(x - mx) for _, x in values]
    return random.choices([i for i, _ in values], weights=weights, k=1)[0]

def main():
    parser = argparse.ArgumentParser(description='Generador autoregresivo en español')
    parser.add_argument('prompt', nargs='*', help='texto inicial en español')
    parser.add_argument('--tokens', type=int, default=100, help='bytes nuevos (default: 100)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='0=greedy; por ejemplo 0.7 activa sampling')
    parser.add_argument('--top-k', type=int, default=0,
                        help='limitar sampling a los k tokens más probables')
    parser.add_argument('--vocab-file', default='data/train.txt',
                        help='corpus que define los bytes permitidos')
    args = parser.parse_args()
    prompt = ' '.join(args.prompt) if args.prompt else input('Prompt: ')
    if not prompt.strip():
        prompt = 'Hola, ¿cómo estás?'
    if args.tokens < 0 or args.tokens > 10000:
        parser.error('--tokens debe estar entre 0 y 10000')
    if args.temperature < 0:
        parser.error('--temperature no puede ser negativa')
    if args.top_k < 0 or args.top_k > VOCAB_SIZE:
        parser.error(f'--top-k debe estar entre 0 y {VOCAB_SIZE}')

    model = TinyTransformer()
    load_model(model, 'models/model.bin')
    with open(args.vocab_file, 'rb') as f:
        corpus_bytes = set(f.read())
    tokens = encode(prompt)
    for _ in range(args.tokens):
        ctx = tokens[-CONTEXT_SIZE:]
        logits = model.forward(ctx)
        last_start = (len(ctx) - 1) * VOCAB_SIZE
        row = logits.data[last_start:last_start + VOCAB_SIZE]
        tokens.append(choose_token(row, tokens, corpus_bytes, args.temperature, args.top_k))
    print('IA:', decode(tokens))

if __name__ == '__main__':
    main()
