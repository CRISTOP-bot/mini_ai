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
    moving=None
    for step in range(1,steps+1):
        xs,ys=data.sample(BATCH_SIZE); opt.zero_grad(); losses=[]
        for x,y in zip(xs,ys):
            loss=cross_entropy(model.forward(x),y); loss.backward(); losses.append(loss.data[0])
        for p in model.params:
            p.grad=[max(-1.0,min(1.0,g/BATCH_SIZE)) for g in p.grad]
        opt.step()
        current=sum(losses)/len(losses); moving=current if moving is None else .95*moving+.05*current
        # Las closures del grafo de autograd forman ciclos; recogerlos evita
        # que la RAM crezca indefinidamente en sesiones largas de Termux.
        if step % 10 == 0:
            gc.collect()
        if step==1 or step%PRINT_EVERY==0:
            print(f'Step {step}/{steps}\nLoss: {current:.4f} | media móvil: {moving:.4f}')
        if step%CHECKPOINT_EVERY==0: save_model(model,'models/model.bin')
    save_model(model,'models/model.bin'); print('Guardado en models/model.bin')
if __name__=='__main__': main()
