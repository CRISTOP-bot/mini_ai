import mmap, random
class TextDataset:
    def __init__(self,path,context_size,seed=42,val_fraction=0.1):
        self.path=path; self.context_size=context_size; self.file=open(path,'rb'); self.mm=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ); self.length=len(self.mm)
        if self.length<context_size+2: self.close(); raise ValueError(f'El dataset necesita al menos {context_size+2} bytes.')
        self.rng=random.Random(seed); starts=list(range(self.length-context_size)); self.rng.shuffle(starts); cut=max(1,int(len(starts)*val_fraction)); self.val_starts=starts[:cut]; self.train_starts=starts[cut:]
    def _at(self,start):
        b=self.mm[start:start+self.context_size+1]; return list(b[:-1]),list(b[1:])
    def sample(self,batch_size=1):
        starts=[self.rng.choice(self.train_starts) for _ in range(batch_size)]
        return tuple(zip(*[self._at(s) for s in starts]))
    def validation(self,n=8): return [self._at(s) for s in self.val_starts[:min(n,len(self.val_starts))]]
    def __len__(self): return self.length
    def close(self):
        if getattr(self,'mm',None) is not None: self.mm.close(); self.mm=None
        if getattr(self,'file',None) is not None: self.file.close(); self.file=None
    def __del__(self):
        try:self.close()
        except Exception:pass
