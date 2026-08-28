from __future__ import annotations
import argparse, copy, json, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

N_STATE=8; N_NOTE=32; NOTE_BASE=N_STATE; QUERY=NOTE_BASE+N_NOTE; ANSWER_BASE=QUERY+1; VOCAB=ANSWER_BASE+N_STATE
DISTANCES=(32,64,128,256); LOCAL_WINDOW=16; D=24; K=6

def base_sequence(batch,max_distance,seed):
    g=torch.Generator(device='cpu').manual_seed(seed); states=torch.randint(0,N_STATE,(batch,),generator=g); motifs=torch.randint(0,8,(batch,),generator=g)
    x=torch.empty((batch,max_distance+2),dtype=torch.long); x[:,0]=states
    for t in range(1,x.shape[1]): x[:,t]=NOTE_BASE+((motifs+t+3*((t//4)%4))%N_NOTE)
    return x,states

def train_batch(batch,max_distance,device,seed):
    x,states=base_sequence(batch,max_distance,seed); g=torch.Generator(device='cpu').manual_seed(seed+17); ds=torch.tensor(DISTANCES); qd=ds[torch.randint(0,len(ds),(batch,),generator=g)]; rows=torch.arange(batch); x[rows,qd]=QUERY; x[rows,qd+1]=ANSWER_BASE+states; inp=x[:,:-1].to(device); return inp,x[:,1:].to(device),inp.eq(QUERY),states.to(device)
def eval_batch(batch,distance,max_distance,device,seed):
    x,states=base_sequence(batch,max_distance,seed); x[:,distance]=QUERY; x[:,distance+1]=ANSWER_BASE+states; inp=x[:,:-1].to(device); return inp,x[:,1:].to(device),inp.eq(QUERY),states.to(device)

def windows(h,window=LOCAL_WINDOW):
    b,t,d=h.shape; hp=torch.cat([h.new_zeros((b,window-1,d)),h],1); w=hp.unfold(1,window,1).permute(0,1,3,2).contiguous().view(b*t,window,d); idx=torch.arange(t,device=h.device); valid=(torch.arange(window,device=h.device)[None,:]>=(window-1-idx[:,None])).repeat(b,1); return w,~valid

class TinyBlock(nn.Module):
    def __init__(self,d=D): super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,4,dropout=0,batch_first=True); self.n2=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,3*d),nn.GELU(),nn.Linear(3*d,d))
    def forward(self,x,kpm=None): n=self.n1(x); a,_=self.attn(n,n,n,key_padding_mask=kpm,need_weights=False); x=x+a; return x+self.ff(self.n2(x))
class Local(nn.Module):
    def __init__(self): super().__init__(); self.block=TinyBlock()
    def forward(self,h): b,t,d=h.shape; w,kpm=windows(h); return self.block(w,kpm)[:,-1].view(b,t,d)
class LinearMemory(nn.Module):
    def __init__(self): super().__init__(); self.q=nn.Linear(D,K,bias=False); self.k=nn.Linear(D,K,bias=False); self.v=nn.Linear(D,D,bias=False); self.write=nn.Linear(D,1); self.mix=nn.Linear(2*D,D); self.norm=nn.LayerNorm(D); self.logit_decay=nn.Parameter(torch.tensor(4.0)); nn.init.constant_(self.write.bias,-1)
    def forward(self,h):
        b,t,d=h.shape; x=self.norm(h); q=F.elu(self.q(x))+1; k=F.elu(self.k(x))+1; v=self.v(x); w=torch.sigmoid(self.write(x)); decay=torch.sigmoid(self.logit_decay).clamp(.9,.9999); c=w[:,:,:,None]*torch.einsum('btk,btd->btkd',k,v); zc=w*k; idx=torch.arange(t,device=h.device,dtype=h.dtype); inv=decay.pow(-idx); f=decay.pow(idx); s=torch.cumsum(c*inv[None,:,None,None],1)*f[None,:,None,None]; z=torch.cumsum(zc*inv[None,:,None],1)*f[None,:,None]; slots=s/(z[:,:,:,None]+1e-5); read=torch.einsum('btk,btkd->btd',q,s)/(torch.einsum('btk,btk->bt',q,z)[:,:,None]+1e-5); return self.mix(torch.cat([h,read],-1)),slots
class MemoryEncoder(nn.Module):
    def __init__(self): super().__init__(); self.emb=nn.Embedding(VOCAB,D); self.memory=LinearMemory(); self.norm=nn.LayerNorm(D); self.state_head=nn.Linear(D,N_STATE); self.recon_head=nn.Linear(D,VOCAB)
    def forward(self,ids): h=self.emb(ids); m,slots=self.memory(h); z=self.norm(h+m); return z,slots,self.state_head(z),self.recon_head(z)

def pretrain_memory(seed,steps,batch,max_distance,device):
    torch.manual_seed(seed); random.seed(seed); m=MemoryEncoder().to(device); opt=torch.optim.AdamW(m.parameters(),lr=4e-3,weight_decay=.01); m.train()
    for step in range(steps):
        x,_,_,states=train_batch(batch,max_distance,device,seed*100000+step); z,slots,sl,recon=m(x); positions=torch.arange(x.shape[1],device=device)>=LOCAL_WINDOW; mask=positions[None,:].expand(x.shape[0],-1); st=states[:,None].expand_as(x); loss_state=F.cross_entropy(sl[mask],st[mask]); loss_recon=F.cross_entropy(recon.reshape(-1,VOCAB),x.reshape(-1)); loss=3.0*loss_state+loss_recon; opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1); opt.step()
    return m

def memory_accuracy(m,device,max_distance):
    m.eval(); out={}
    with torch.no_grad():
      for d in DISTANCES:
        x,y,q,s=eval_batch(64,d,max_distance,device,80000+d); z,slots,sl,recon=m(x); pos=torch.arange(x.shape[1],device=device)>=d; mask=pos[None,:].expand(x.shape[0],-1); st=s[:,None].expand_as(x); out[d]={'state_accuracy_after_distance':float((sl.argmax(-1)[mask]==st[mask]).float().mean()),'reconstruction_accuracy':float((recon.argmax(-1)==x).float().mean())}
    return out

class DownstreamBase(nn.Module):
    def __init__(self,memory): super().__init__(); self.memory=memory; [p.requires_grad_(False) for p in self.memory.parameters()]; self.local=Local(); self.norm=nn.LayerNorm(D); self.head=nn.Linear(D,VOCAB)
    def parts(self,ids):
        with torch.no_grad(): mem,slots,_,_=self.memory(ids)
        return mem,slots,self.local(mem)
class FixedResidual(DownstreamBase):
    def forward(self,ids): mem,slots,loc=self.parts(ids); return self.head(self.norm(loc+mem))
class GatedResidual(DownstreamBase):
    def __init__(self,m): super().__init__(m); self.proj=nn.Linear(D,D,bias=False); self.gate=nn.Linear(2*D,D)
    def forward(self,ids): mem,slots,loc=self.parts(ids); g=torch.sigmoid(self.gate(torch.cat([loc,mem],-1))); return self.head(self.norm(loc+g*self.proj(mem)))
class CrossAttention(DownstreamBase):
    def __init__(self,m): super().__init__(m); self.cross=nn.MultiheadAttention(D,4,dropout=0,batch_first=True)
    def forward(self,ids): mem,slots,loc=self.parts(ids); b,t,k,d=slots.shape; c,_=self.cross(loc.reshape(b*t,1,d),slots.reshape(b*t,k,d),slots.reshape(b*t,k,d),need_weights=False); return self.head(self.norm(loc+c.reshape(b,t,d)))
class MemoryTokens(DownstreamBase):
    def __init__(self,m): super().__init__(m); self.fuser=TinyBlock()
    def forward(self,ids): mem,slots,loc=self.parts(ids); b,t,k,d=slots.shape; z=torch.cat([slots,loc[:,:,None,:]],2).reshape(b*t,k+1,d); return self.head(self.norm(self.fuser(z)[:,-1].view(b,t,d)))
MODELS={'fixed_residual':FixedResidual,'gated_residual':GatedResidual,'cross_attention':CrossAttention,'memory_tokens':MemoryTokens}
@dataclass
class Metric: distance:int; query_accuracy:float; local_accuracy:float; val_loss:float
@dataclass
class Result: model:str; seed:int; trainable_parameters:int; total_parameters:int; ms_per_step:float; memory_pretrain:dict; metrics:list[Metric]

def evaluate(model,device,max_distance):
    model.eval(); out=[]
    with torch.no_grad():
      for d in DISTANCES:
        qok=qn=lok=ln=0; losses=[]
        for i in range(2):
          x,y,q,s=eval_batch(32,d,max_distance,device,90000+d*7+i); logits=model(x); losses.append(float(F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1)))); p=logits.argmax(-1); lm=~q; qok+=int(((p==y)&q).sum()); qn+=int(q.sum()); lok+=int(((p==y)&lm).sum()); ln+=int(lm.sum())
        out.append(Metric(d,qok/qn,lok/ln,sum(losses)/len(losses)))
    return out

def run(name,seed,memory_steps,down_steps,batch,max_distance,device):
    memory=pretrain_memory(seed,memory_steps,batch,max_distance,device); ma=memory_accuracy(memory,device,max_distance); torch.manual_seed(seed+1000); model=MODELS[name](memory).to(device); opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-3,weight_decay=.01); t0=time.perf_counter(); model.train()
    for step in range(down_steps):
      x,y,q,s=train_batch(batch,max_distance,device,seed*200000+step); logits=model(x); raw=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1),reduction='none').view_as(y); w=torch.ones_like(raw); w[q]=32; loss=(raw*w).sum()/w.sum(); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1); opt.step()
    elapsed=time.perf_counter()-t0; return Result(name,seed,sum(p.numel() for p in model.parameters() if p.requires_grad),sum(p.numel() for p in model.parameters()),elapsed*1000/down_steps,ma,evaluate(model,device,max_distance))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=MODELS,required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--memory-steps',type=int,default=120); ap.add_argument('--down-steps',type=int,default=100); ap.add_argument('--batch',type=int,default=6); ap.add_argument('--max-distance',type=int,default=256); ap.add_argument('--out',required=True); a=ap.parse_args(); torch.set_num_threads(min(4,torch.get_num_threads())); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); r=run(a.model,a.seed,a.memory_steps,a.down_steps,a.batch,a.max_distance,device); payload={'schema_version':3,'device':str(device),'task':{'distances':list(DISTANCES),'local_window':LOCAL_WINDOW,'query_chance':1/N_STATE,'training':'stage1 memory state+reconstruction then freeze; stage2 local generator'},'result':{**asdict(r),'metrics':[asdict(m) for m in r.metrics]}}; Path(a.out).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=='__main__': main()
