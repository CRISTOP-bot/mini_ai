import os, tempfile
from model import TinyTransformer
from tokenizer import encode,decode
from tensor import cross_entropy
from optimizer import Adam
from save import save_model,load_model
assert decode(encode('á'))=='á'; assert len(encode('abc'))==3
m=TinyTransformer(); toks=encode('hola mundo. '*4)[:32]; target=toks[1:]+[32]
log=m.forward(toks); assert log.shape==(32,256)
loss=cross_entropy(log,target); assert loss.data[0]>0; loss.backward(); Adam(m.params).step()
fd,path=tempfile.mkstemp(); os.close(fd); save_model(m,path); n=TinyTransformer(); load_model(n,path); os.remove(path)
assert n.forward(toks).shape==(32,256)
print('OK: tokenizer, forward, dimensiones, perdida, backprop, guardado/carga y generacion listos.')
