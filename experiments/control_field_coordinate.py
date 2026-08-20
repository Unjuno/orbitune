import math, random, statistics, time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(5)
SEED=20260820
random.seed(SEED); torch.manual_seed(SEED)


def basis(x, K=8):
    ks=torch.arange(K, device=x.device, dtype=x.dtype)
    return torch.cos(math.pi*x[...,None]*ks)


class Field(nn.Module):
    def __init__(self, nctrl=4, emb=32, K=8, channels=3):
        super().__init__(); self.K=K; self.channels=channels
        self.emb=nn.Embedding(nctrl,emb)
        self.mlp=nn.Sequential(nn.Linear(emb,64),nn.GELU(),nn.Linear(64,K*channels))
    def forward(self, ctrl, coord):
        co=self.mlp(self.emb(ctrl)).view(-1,self.K,self.channels)
        y=torch.einsum('btk,bkc->btc',basis(coord,self.K),co)
        return y*(ctrl!=0).float()[:,None,None]


@dataclass
class Batch:
    token_coord: torch.Tensor
    music_coord: torch.Tensor
    ctrl: torch.Tensor
    target: torch.Tensor


def make_batch(B=32,T=96, device='cpu'):
    token_coord=torch.linspace(0,1,T,device=device)[None,:].repeat(B,1)
    music=[]; ctrls=[]; targets=[]
    for _ in range(B):
        ctrl=random.choice([1,2,3])
        base=random.uniform(0.5,2.0); phase=random.uniform(0,2*math.pi); dts=[]
        for i in range(T):
            u=i/(T-1)
            density=max(0.2,base*(1.0+0.75*math.sin(2*math.pi*u+phase)))
            dts.append((1.0/density)*random.uniform(0.7,1.3))
        cum=torch.tensor(dts,device=device).cumsum(0)
        m=(cum-cum[0])/(cum[-1]-cum[0]+1e-8)
        music.append(m)
        if ctrl==1:
            y=torch.stack([1-m,m,m],-1)
        elif ctrl==2:
            y=torch.stack([0.5+0.45*torch.sin(2*math.pi*m),0.5+0.45*torch.sin(2*math.pi*m+math.pi),0.5+0.45*torch.sin(2*math.pi*m-math.pi/2)],-1)
        else:
            hump=torch.sin(math.pi*m).clamp_min(0)
            y=torch.stack([1-0.6*hump,hump,0.2+0.7*hump],-1)
        ctrls.append(ctrl); targets.append(y)
    return Batch(token_coord,torch.stack(music),torch.tensor(ctrls,device=device),torch.stack(targets))


def train(mode, steps=600):
    model=Field(); opt=torch.optim.AdamW(model.parameters(),lr=3e-3); losses=[]; t0=time.time()
    for _ in range(steps):
        bt=make_batch(B=24,T=96); coord=bt.music_coord if mode=='music' else bt.token_coord
        loss=F.mse_loss(model(bt.ctrl,coord),bt.target)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    return model,losses,time.time()-t0


def corr(a,b):
    a=a.flatten()-a.mean(); b=b.flatten()-b.mean()
    return float((a*b).sum()/((a.square().sum()*b.square().sum()).sqrt()+1e-12))


def evaluate(model,mode,batches=50):
    mse=[]; cors=[]; phase=[]
    with torch.no_grad():
        for _ in range(batches):
            bt=make_batch(B=32,T=96); coord=bt.music_coord if mode=='music' else bt.token_coord; p=model(bt.ctrl,coord)
            mse.append(F.mse_loss(p,bt.target).item())
            for b in range(p.size(0)):
                cors.append(corr(p[b],bt.target[b]))
                if int(bt.ctrl[b])==2: phase.append(corr(p[b,:,0],bt.target[b,:,0]))
    return {'mse':statistics.mean(mse),'corr':statistics.mean(cors),'wave_corr':statistics.mean(phase)}


if __name__ == '__main__':
    for mode in ['token','music']:
        model,losses,secs=train(mode); ev=evaluate(model,mode)
        print(mode,'params',sum(p.numel() for p in model.parameters()),'sec',round(secs,2),'start',losses[0],'end',losses[-1],ev)
    m=Field(); coord=torch.rand(5,96); ctrl=torch.zeros(5,dtype=torch.long)
    print('none_max',float(m(ctrl,coord).abs().max().detach()))
