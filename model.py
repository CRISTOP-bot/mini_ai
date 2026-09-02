import random, math
from tensor import Tensor, parameter, add, matmul, tanh
from config import *

class TinyTransformer:
    """Una capa transformer pequeña, con NUM_HEADS cabezas de atención causal."""
    def __init__(self, seed=SEED):
        self.rng=random.Random(seed); d=EMBEDDING_DIM
        if d % NUM_HEADS: raise ValueError('EMBEDDING_DIM debe ser divisible por NUM_HEADS')
        self.params=[]
        self.E=self._p((VOCAB_SIZE,d)); self.P=self._p((CONTEXT_SIZE,d))
        self.Wq=self._p((d,d)); self.Wk=self._p((d,d)); self.Wv=self._p((d,d)); self.Wo=self._p((d,d))
        self.W1=self._p((d,HIDDEN_DIM)); self.W2=self._p((HIDDEN_DIM,d)); self.Wout=self._p((d,VOCAB_SIZE))
    def _p(self,s):
        p=parameter(s,self.rng); self.params.append(p); return p
    def embed(self,tokens):
        T=len(tokens); d=EMBEDDING_DIM; vals=[]
        for t in tokens: vals += self.E.data[t*d:(t+1)*d]
        out=Tensor(vals,(T,d),True,(self.E,),'gather')
        def back():
            for i,t in enumerate(tokens):
                for j in range(d): self.E.grad[t*d+j]+=out.grad[i*d+j]
        out._backward=back; return out
    def positional(self,T):
        d=EMBEDDING_DIM
        out=Tensor([self.P.data[i*d+j] for i in range(T) for j in range(d)],(T,d),True,(self.P,),'pos')
        def back():
            for i in range(T):
                for j in range(d): self.P.grad[i*d+j]+=out.grad[i*d+j]
        out._backward=back; return out
    def attention(self,q,k,v,T):
        """Atención multi-cabeza con softmax causal y backward explícito."""
        h,dh=NUM_HEADS,HEAD_DIM; probs=[]; scores=[]
        for head in range(h):
            sc=[]
            for i in range(T):
                raw=[sum(q.data[i*EMBEDDING_DIM+head*dh+r]*k.data[j*EMBEDDING_DIM+head*dh+r] for r in range(dh))/math.sqrt(dh) if j<=i else -1e9 for j in range(T)]
                mx=max(raw); ex=[math.exp(z-mx) for z in raw]; z=sum(ex); sc.append([e/z for e in ex])
            probs.append(sc)
        vals=[]
        for i in range(T):
            for head in range(h):
                for r in range(dh): vals.append(sum(probs[head][i][j]*v.data[j*EMBEDDING_DIM+head*dh+r] for j in range(T)))
        out=Tensor(vals,(T,EMBEDDING_DIM),q.requires_grad or k.requires_grad or v.requires_grad,(q,k,v),'mha')
        def back():
            gq=[0.0]*len(q.data); gk=[0.0]*len(k.data); gv=[0.0]*len(v.data)
            for head in range(h):
                for i in range(T):
                    gp=[0.0]*T
                    for j in range(T):
                        for r in range(dh):
                            go=out.grad[i*EMBEDDING_DIM+head*dh+r]
                            gv[j*EMBEDDING_DIM+head*dh+r]+=probs[head][i][j]*go
                            gp[j]+=v.data[j*EMBEDDING_DIM+head*dh+r]*go
                    dot=sum(probs[head][i][j]*gp[j] for j in range(T))
                    for j in range(i+1):
                        gs=probs[head][i][j]*(gp[j]-dot)/math.sqrt(dh)
                        for r in range(dh):
                            gq[i*EMBEDDING_DIM+head*dh+r]+=gs*k.data[j*EMBEDDING_DIM+head*dh+r]
                            gk[j*EMBEDDING_DIM+head*dh+r]+=gs*q.data[i*EMBEDDING_DIM+head*dh+r]
            q._acc(gq); k._acc(gk); v._acc(gv)
        out._backward=back; return out
    def forward(self,tokens):
        T=len(tokens); x=add(self.embed(tokens),self.positional(T))
        q=matmul(x,self.Wq); k=matmul(x,self.Wk); v=matmul(x,self.Wv)
        x=add(x,matmul(self.attention(q,k,v,T),self.Wo))
        x=add(x,matmul(tanh(matmul(x,self.W1)),self.W2))
        return matmul(x,self.Wout)
    def state_dict(self): return [p.data[:] for p in self.params]
    def load_state_dict(self,state):
        if len(state)!=len(self.params): raise ValueError('checkpoint incompatible')
        for p,d in zip(self.params,state): p.data=d[:]
