"""Autograd diminuto sobre listas planas de Python. No usa NumPy."""
import math

class Tensor:
    def __init__(self, data, shape=None, requires_grad=False, _children=(), _op=''):
        if shape is None:
            if isinstance(data, (int, float)): data = [float(data)]; shape = (1,)
            elif data and isinstance(data[0], list):
                shape = (len(data), len(data[0])); data = [float(x) for row in data for x in row]
            else: data = [float(x) for x in data]; shape = (len(data),)
        self.data = [float(x) for x in data]; self.shape = tuple(shape)
        self.requires_grad = requires_grad; self.grad = [0.0] * len(self.data)
        self._prev = set(_children); self._op = _op; self._backward = lambda: None

    def zero_grad(self): self.grad = [0.0] * len(self.data)
    def _acc(self, g):
        if self.requires_grad:
            for i, x in enumerate(g): self.grad[i] += x
    def backward(self):
        if len(self.data) != 1: raise ValueError('backward() requiere un escalar')
        topo, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev: visit(p)
                topo.append(v)
        visit(self); self.grad[0] = 1.0
        for v in reversed(topo): v._backward()

    def reshape(self, shape):
        if math.prod(shape) != len(self.data): raise ValueError('reshape incompatible')
        out = Tensor(self.data[:], shape, self.requires_grad, (self,), 'reshape')
        def back(): self._acc(out.grad)
        out._backward = back; return out

def parameter(shape, rng):
    scale = math.sqrt(2.0 / shape[-1])
    return Tensor([rng.gauss(0, scale) for _ in range(math.prod(shape))], shape, True)

def add(a, b):
    if isinstance(b, (int,float)): b=Tensor([b], (1,))
    if a.shape != b.shape and b.shape != (1,): raise ValueError(f'add: {a.shape} vs {b.shape}')
    out=Tensor([x+(b.data[0] if b.shape==(1,) else b.data[i]) for i,x in enumerate(a.data)], a.shape, a.requires_grad or b.requires_grad, (a,b), 'add')
    def back():
        a._acc(out.grad)
        if b.shape==(1,): b._acc([sum(out.grad)])
        else: b._acc(out.grad)
    out._backward=back; return out

def mul(a,b):
    if isinstance(b,(int,float)): b=Tensor([b],(1,))
    out=Tensor([x*(b.data[0] if b.shape==(1,) else b.data[i]) for i,x in enumerate(a.data)], a.shape, a.requires_grad or b.requires_grad,(a,b),'mul')
    def back():
        if a.requires_grad: a._acc([g*(b.data[0] if b.shape==(1,) else b.data[i]) for i,g in enumerate(out.grad)])
        if b.requires_grad:
            if b.shape==(1,): b._acc([sum(g*x for g,x in zip(out.grad,a.data))])
            else: b._acc([g*x for g,x in zip(out.grad,a.data)])
    out._backward=back; return out

def matmul(a,b):
    m,k=a.shape; k2,n=b.shape
    if k!=k2: raise ValueError('matmul dimensiones')
    out=Tensor([sum(a.data[i*k+t]*b.data[t*n+j] for t in range(k)) for i in range(m) for j in range(n)],(m,n),a.requires_grad or b.requires_grad,(a,b),'matmul')
    def back():
        if a.requires_grad: a._acc([sum(out.grad[i*n+j]*b.data[t*n+j] for j in range(n)) for i in range(m) for t in range(k)])
        if b.requires_grad: b._acc([sum(a.data[i*k+t]*out.grad[i*n+j] for i in range(m)) for t in range(k) for j in range(n)])
    out._backward=back; return out

def tanh(a):
    vals=[math.tanh(x) for x in a.data]; out=Tensor(vals,a.shape,a.requires_grad,(a,),'tanh')
    out._backward=lambda: a._acc([g*(1-v*v) for g,v in zip(out.grad,vals)]); return out

def relu(a):
    vals=[max(0,x) for x in a.data]; out=Tensor(vals,a.shape,a.requires_grad,(a,),'relu')
    out._backward=lambda: a._acc([g if x>0 else 0 for g,x in zip(out.grad,a.data)]); return out

def softmax_rows(a, causal=False):
    T,N=a.shape; vals=[]
    for i in range(T):
        row=a.data[i*N:(i+1)*N]; mx=max(row); ex=[math.exp(x-mx) for x in row]; z=sum(ex); vals += [x/z for x in ex]
    out=Tensor(vals,a.shape,a.requires_grad,(a,),'softmax')
    def back():
        ga=[0.0]*len(a.data)
        for i in range(T):
            p=vals[i*N:(i+1)*N]; g=out.grad[i*N:(i+1)*N]; dot=sum(x*y for x,y in zip(p,g))
            for j in range(N): ga[i*N+j]=p[j]*(g[j]-dot)
        a._acc(ga)
    out._backward=back; return out

def cross_entropy(logits, targets):
    T,V=logits.shape; probs=[]; loss=0.0
    for i in range(T):
        row=logits.data[i*V:(i+1)*V]; mx=max(row); ex=[math.exp(x-mx) for x in row]; z=sum(ex); p=[x/z for x in ex]; probs+=p; loss-=math.log(max(p[int(targets[i])],1e-12))
    out=Tensor([loss/T],(1,),logits.requires_grad,(logits,),'cross_entropy')
    def back():
        g=[x/T for x in probs]
        for i,t in enumerate(targets): g[i*V+int(t)]-=1.0/T
        logits._acc([x*out.grad[0] for x in g])
    out._backward=back; return out
