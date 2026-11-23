import numpy as np
runs={'L none':'results/toy/bal/c2k8-normal-s0','L tag-norm':'results/toy/place/c2k8-normal-pv-s0','L tag-rev':'results/toy/place/c2k8-rev-pv-s0',
'S none':'results/toy/bal/c2k1-normal-s0','S tag-norm':'results/toy/place/c2k1-normal-pv-s0','S tag-rev':'results/toy/place/c2k1-rev-pv-s0'}
rng=np.random.default_rng(0)
def load(p,K):
    d=np.load(p+'/per_example.npz'); m=d['K']==K; return {k:d[k][m] for k in d.files}
def acc(c):
    if len(c)==0: return float('nan'),0
    return c.mean(),len(c)
D={n:load(p,8 if n[0]=='L' else 1) for n,p in runs.items()}
def metrics(d):
    o={}
    m=(d['op_digits']==1)&(d['ans_digits']>=3)&(d['ans_digits']<=5)&(d['depth']>=2)&(d['depth']<=6); o['big,1d-in,d2-6']=m
    m=(d['op_digits']==1)&(d['ans_digits']<=2)&(d['depth']>=2)&(d['depth']<=6); o['small,1d-in,d2-6']=m
    m=(d['op_digits']==1)&(d['depth']<=3); o['all 1d-in d1-3']=m
    m=(d['op_digits']>=2)&(d['depth']<=3); o['multi-in d1-3']=m
    for dep in (1,2,3):
        for od in (2,3):
            m=(d['op_digits']==od)&(d['depth']==dep)
            if m.sum(): o[f'in{od}d depth{dep}']=m
    for dep in (8,10,13):
        o[f'1d-ans depth{dep}']=(d['ans_digits']==1)&(d['depth']==dep)&(d['op_digits']==1)
    return o
names=list(runs)
rows=None
print('%-20s'%'metric'+''.join('%12s'%n for n in names))
for key in metrics(D[names[0]]):
    line='%-20s'%key
    for n in names:
        a,c=acc(D[n]['correct'][metrics(D[n])[key]]); line+='%8.3f/%-4d'%(a,c//100) if c else '%12s'%'-'
    print(line)
# paired bootstrap diff looped tag-rev - tag-norm, tag-norm - none
def boot(a,b,B=2000):
    da=a.astype(float); db=b.astype(float)
    s=[rng.choice(da,len(da)).mean()-rng.choice(db,len(db)).mean() for _ in range(B)]
    return da.mean()-db.mean(),np.percentile(s,2.5),np.percentile(s,97.5)
for A,Bn in [('L tag-norm','L none'),('L tag-rev','L none'),('L tag-rev','L tag-norm')]:
    for key in ['big,1d-in,d2-6','small,1d-in,d2-6','all 1d-in d1-3','multi-in d1-3']:
        r=boot(D[A]['correct'][metrics(D[A])[key]],D[Bn]['correct'][metrics(D[Bn])[key]])
        print(f'{A} - {Bn} {key}: {r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]')
