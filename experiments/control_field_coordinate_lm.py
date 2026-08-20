import math, random, statistics, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(5); random.seed(7); torch.manual_seed(7)
V=96

def basis(x,K=8):
    ks=torch.arange(K,device=x.device,dtype=x.dtype); return torch.cos(math.pi*x[...,None]*ks)

class Field(nn.Module):
    def __init__(self,d=448,K=8):
        super().__init__(); self.K=K
        self.emb=nn.Embedding(4,32); self.mlp=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,K*3)); self.tofilm=nn.Linear(3,2*d)
    def forward(self,ctrl,coord):
        co=self.mlp(self.emb(ctrl)).view(-1,self.K,3)
        c=torch.einsum('btk,bkc->btc',basis(coord,self.K),co)*(ctrl!=0).float()[:,None,None]
        g,b=self.tofilm(c).chunk(2,-1); return c,g,b

class Attn(nn.Module):
    def __init__(self,d=448,h=7):
        super().__init__(); self.h=h; self.hd=d//h
        self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,d,bias=False); self.v=nn.Linear(d,d,bias=False); self.o=nn.Linear(d,d,bias=False)
    def forward(self,x):
        B,T,D=x.shape
        q=self.q(x).view(B,T,self.h,self.hd).transpose(1,2); k=self.k(x).view(B,T,self.h,self.hd).transpose(1,2); v=self.v(x).view(B,T,self.h,self.hd).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.o(y.transpose(1,2).reshape(B,T,D))

class Block(nn.Module):
    def __init__(self,d=448,h=7):
        super().__init__(); self.l1=nn.LayerNorm(d); self.a=Attn(d,h); self.l2=nn.LayerNorm(d); self.m=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x,g,b):
        z=(1+0.1*torch.tanh(g))*self.l1(x)+0.1*b; x=x+self.a(z)
        z=(1+0.1*torch.tanh(g))*self.l2(x)+0.1*b; return x+self.m(z)

class Model(nn.Module):
    def __init__(self):
        super().__init__(); d=448
        self.e=nn.Embedding(V,d); self.p=nn.Embedding(1024,d); self.f=Field(d); self.bs=nn.ModuleList([Block(d,7) for _ in range(4)]); self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,V,bias=False); self.head.weight=self.e.weight
    def forward(self,x,ctrl,coord,force_none=False):
        B,T=x.shape; z=self.e(x)+self.p(torch.arange(T,device=x.device))[None]
        c,g,b=self.f(torch.zeros_like(ctrl) if force_none else ctrl,coord)
        for bl in self.bs: z=bl(z,g,b)
        return self.head(self.ln(z)),c

def batch(B=4):
    xs=[]; coords=[]; ctrls=[]
    for _ in range(B):
        ctrl=random.choice([1,2,3]); ctrls.append(ctrl); phase=random.random()*2*math.pi; base=random.uniform(.5,2.0); dts=[]
        for i in range(16):
            u=i/15; dens=max(.2,base*(1+.75*math.sin(2*math.pi*u+phase))); dts.append((1/dens)*random.uniform(.75,1.25))
        mt=torch.tensor(dts).cumsum(0); mt=(mt-mt[0])/(mt[-1]-mt[0]+1e-8); toks=[]; c=[]
        for m in mt.tolist():
            if ctrl==1: f=0.2
            elif ctrl==2: f=m
            else: f=.5+.45*math.sin(2*math.pi*m)
            timebin=max(0,min(15,int(round((1-f)*15)))); rest_prob=.55*(1-f)
            pitch=16+(31 if random.random()<rest_prob else int(8+20*f+random.gauss(0,2))); pitch=max(16,min(47,pitch))
            vel=48+max(0,min(31,int(round(6+25*f+random.gauss(0,2))))); toks += [timebin,pitch,vel]; c += [m,m,m]
        xs.append(toks); coords.append(c)
    x=torch.tensor(xs); return x[:,:-1],x[:,1:],torch.tensor(ctrls),torch.tensor(coords)[:,:-1]

def train(mode,steps=45):
    model=Model(); opt=torch.optim.AdamW(model.parameters(),lr=7e-4); ls=[]; t0=time.time()
    for _ in range(steps):
        x,y,ctrl,mc=batch(); coord=mc if mode=='music' else torch.linspace(0,1,x.size(1))[None].repeat(x.size(0),1)
        logits,_=model(x,ctrl,coord); loss=F.cross_entropy(logits.reshape(-1,V),y.reshape(-1)); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); ls.append(loss.item())
    return model,ls,time.time()-t0

@torch.no_grad()
def evaluate(model,mode,n=24):
    on=[]; off=[]
    for _ in range(n):
        x,y,ctrl,mc=batch(); coord=mc if mode=='music' else torch.linspace(0,1,x.size(1))[None].repeat(x.size(0),1)
        l,_=model(x,ctrl,coord); ln,_=model(x,ctrl,coord,True)
        on.append(F.cross_entropy(l.reshape(-1,V),y.reshape(-1)).item()); off.append(F.cross_entropy(ln.reshape(-1,V),y.reshape(-1)).item())
    return statistics.mean(on),statistics.mean(off)

if __name__ == '__main__':
    for mode in ['token','music']:
        m,ls,sec=train(mode); on,off=evaluate(m,mode)
        print(mode,'params',sum(p.numel() for p in m.parameters()),'sec',round(sec,2),'start',round(ls[0],4),'end',round(ls[-1],4),'heldout',round(on,4),'control_off',round(off,4),'delta',round(off-on,4))
