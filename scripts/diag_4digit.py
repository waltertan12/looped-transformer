import sys, numpy as np, torch
sys.path.insert(0,'scripts')
from eval_ckpt import load_checkpoint, train_k
from looptf.toy.data import encode, decode, STOI
torch.manual_seed(0)
rng=np.random.default_rng(0)
def greedy(model,k,prompt,rev,maxn=9):
    x=encode(prompt); out=''
    for _ in range(maxn):
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            lg=model(torch.tensor([x],device='cuda'),n_loops=k)
        t=int(lg[0,-1].argmax()); x.append(t); c=decode([t]); 
        if c=='#': break
        out+=c
    return out
cases=[]
for nd in (3,4):
    for _ in range(300):
        a=int(rng.integers(10**(nd-1),10**nd)); b=int(rng.integers(10**(nd-1),10**nd)); cases.append((nd,a,b))
for ck in ['bal/c2k8-normal-s0','bal/c16k1-normal-s0','long/c16k1-s0','long/c16k1-rev-s0']:
    m,args=load_checkpoint(f'checkpoints/toy/{ck}.pt','cuda'); k=train_k(args); rev=args.get('reverse_answer',False)
    res={3:[],4:[]}; ex=[]
    for nd,a,b in cases:
        o=greedy(m,k,f'{a}+{b}=',rev)
        truth=str(a+b); 
        pred=o if not rev else (o[::-1])
        res[nd].append(pred==truth)
        if nd==4 and len(ex)<6: ex.append(f'{a}+{b}={truth} -> {pred}')
    print(ck,'tags' if args.get('place_emb') else 'no-tags','3d acc %.2f'%np.mean(res[3]),'4d acc %.2f'%np.mean(res[4]))
    for e in ex: print('   ',e)
print('--- 4-digit a+b: length right / leading 2 digits right / within 5%')
for ck in ['bal/c2k8-normal-s0','bal/c16k1-normal-s0','long/c2k8-s0','long/c16k1-s0']:
    m,args=load_checkpoint(f'checkpoints/toy/{ck}.pt','cuda'); k=train_k(args)
    L=P=W=0; n=0
    for nd,a,b in cases:
        if nd!=4: continue
        o=greedy(m,k,f'{a}+{b}=',False); t=str(a+b); n+=1
        L+=len(o)==len(t); P+=o[:2]==t[:2]
        try: W+=abs(int(o)-(a+b))<=0.05*(a+b)
        except: pass
    print(ck,'tags' if args.get('place_emb') else 'no-tags', f'len {L/n:.2f}  lead2 {P/n:.2f}  within5% {W/n:.2f}')
