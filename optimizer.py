class Adam:
    def __init__(self, params, lr=0.003, b1=.9, b2=.999, eps=1e-8):
        self.params=params; self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.t=0
        self.m=[[0.0]*len(p.data) for p in params]; self.v=[[0.0]*len(p.data) for p in params]
    def step(self):
        self.t+=1
        for pi,p in enumerate(self.params):
            for i,g in enumerate(p.grad):
                self.m[pi][i]=self.b1*self.m[pi][i]+(1-self.b1)*g; self.v[pi][i]=self.b2*self.v[pi][i]+(1-self.b2)*g*g
                mh=self.m[pi][i]/(1-self.b1**self.t); vh=self.v[pi][i]/(1-self.b2**self.t)
                p.data[i]-=self.lr*mh/(vh**.5+self.eps)
    def zero_grad(self):
        for p in self.params: p.zero_grad()
