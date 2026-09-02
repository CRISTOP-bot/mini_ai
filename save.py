import pickle, os
from config import VOCAB_SIZE, CONTEXT_SIZE, EMBEDDING_DIM, NUM_HEADS, HIDDEN_DIM

CONFIG_SIGNATURE = {
    'vocab_size': VOCAB_SIZE, 'context_size': CONTEXT_SIZE,
    'embedding_dim': EMBEDDING_DIM, 'num_heads': NUM_HEADS,
    'hidden_dim': HIDDEN_DIM,
}

def save_model(model, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    payload = {'version': 2, 'config': CONFIG_SIGNATURE,
               'state': model.state_dict()}
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=4)

def load_model(model, path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    # Los checkpoints antiguos eran directamente {'state': ...}.
    saved_config = obj.get('config')
    if saved_config is not None and saved_config != CONFIG_SIGNATURE:
        raise ValueError(
            'El checkpoint usa otra configuración. Borra models/model.bin y reentrena.'
        )
    state = obj['state']
    model.load_state_dict(state)
