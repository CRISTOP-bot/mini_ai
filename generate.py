import sys, math
from config import *
from tokenizer import encode,decode
from model import TinyTransformer
from save import load_model

def main():
    prompt=' '.join(sys.argv[1:]) if len(sys.argv)>1 else input('Prompt: ')
    if not prompt.strip(): prompt='hola'
    model=TinyTransformer(); load_model(model,'models/model.bin'); tokens=encode(prompt); new=0
    for _ in range(100):
        ctx=tokens[-CONTEXT_SIZE:]; logits=model.forward(ctx)
        # Seleccionamos solo la fila correspondiente al último token.
        last_start=(len(ctx)-1)*VOCAB_SIZE
        row=logits.data[last_start:last_start+VOCAB_SIZE]
        # greedy: estable, barato y fácil de estudiar
        token=max(range(VOCAB_SIZE),key=lambda i:row[i]); tokens.append(token); new+=1
    print('IA:',decode(tokens))
if __name__=='__main__': main()
