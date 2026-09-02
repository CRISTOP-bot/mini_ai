import sys, os, math, gc
from config import *
from dataset import TextDataset
from model import TinyTransformer
from tensor import cross_entropy
from optimizer import Adam
from save import save_model

def main():
    path=sys.argv[1] if len(sys.argv)>1 else 'data/train.txt'; steps=int(os.getenv('STEPS',DEFAULT_STEPS))
    data=TextDataset(path,CONTEXT_SIZE); model=TinyTransformer(); opt=Adam(model.params,LEARNING_RATE,BETA1,BETA2,EPSILON)
    print(f'Dataset: {len(data)} bytes | parámetros: {sum(len(p.data) for p in model.params):,}')
    for step in range(1,steps+1):
        xs,ys=data.sample(1); opt.zero_grad(); logits=model.forward(xs[0]); loss=cross_entropy(logits,ys[0]); loss.backward()
        for p in model.params: p.grad=[max(-1.0,min(1.0,g)) for g in p.grad]
        opt.step()
        # Las closures del grafo de autograd forman ciclos; recogerlos evita
        # que la RAM crezca indefinidamente en sesiones largas de Termux.
        if step % 10 == 0:
            gc.collect()
        if step==1 or step%PRINT_EVERY==0: print(f'Step {step}/{steps}\nLoss: {loss.data[0]:.4f}')
        if step%CHECKPOINT_EVERY==0: save_model(model,'models/model.bin')
    save_model(model,'models/model.bin'); print('Guardado en models/model.bin')
if __name__=='__main__': main()
