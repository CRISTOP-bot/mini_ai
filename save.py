import pickle, os
from config import VOCAB_SIZE, CONTEXT_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM
CONFIG_SIGNATURE={'vocab_size':VOCAB_SIZE,'context_size':CONTEXT_SIZE,'embedding_dim':EMBEDDING_DIM,'num_heads':NUM_HEADS,'hidden_dim':HIDDEN_DIM}
def save_checkpoint(model, optimizer, step, early, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    obj={'version':3,'config':CONFIG_SIGNATURE,'state':model.state_dict(),'optimizer':{'t':optimizer.t,'m':optimizer.m,'v':optimizer.v},'step':step,'best_val_loss':early.best,'bad_evals':early.bad_evals}
    with open(path,'wb') as f: pickle.dump(obj,f,protocol=4)
def load_checkpoint(model, optimizer, early, path):
    with open(path,'rb') as f: obj=pickle.load(f)
    if obj.get('config') and obj['config']!=CONFIG_SIGNATURE: raise ValueError('Checkpoint incompatible: borra y reentrena.')
    model.load_state_dict(obj['state'])
    if 'optimizer' in obj:
        optimizer.t=obj['optimizer']['t']; optimizer.m=obj['optimizer']['m']; optimizer.v=obj['optimizer']['v']
    if 'best_val_loss' in obj: early.best=obj['best_val_loss']
    early.bad_evals=obj.get('bad_evals',0)
    return obj.get('step',0)
def save_model(model,path):
    class E: best=float('inf'); bad_evals=0
    class O: t=0; m=[]; v=[]
    save_checkpoint(model,O(),0,E(),path)
def load_model(model,path):
    class E: best=float('inf'); bad_evals=0
    class O: t=0; m=[]; v=[]
    load_checkpoint(model,O(),E(),path)
