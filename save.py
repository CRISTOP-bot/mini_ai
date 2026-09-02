import pickle, os

def save_model(model, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path,'wb') as f: pickle.dump({'state':model.state_dict()},f,protocol=4)

def load_model(model,path):
    with open(path,'rb') as f: obj=pickle.load(f)
    model.load_state_dict(obj['state'])
