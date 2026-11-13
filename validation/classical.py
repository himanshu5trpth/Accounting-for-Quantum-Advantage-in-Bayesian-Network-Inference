"""Exact discrete-proposal simulation for eligible frozen validation cases."""
from pathlib import Path
import gzip,json,math,re,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'));from msbn import MSBN,prune

def public_bn(name):
 import pgmpy;text=gzip.open(Path(pgmpy.__file__).parent/'utils'/'example_models'/f'{name}.bif.gz','rt').read()
 states={}
 for node,n,ss in re.findall(r'variable\s+([^\s{]+)\s*\{\s*type\s+discrete\s*\[\s*(\d+)\s*\]\s*\{([^}]+)\}',text,re.S):
  states[node.strip('"')]=[s.strip().strip('"') for s in ss.split(',')]
 specs={}
 for desc,body in re.findall(r'probability\s*\((.*?)\)\s*\{(.*?)\}',text,re.S):
  d=desc.split('|');node=d[0].strip().strip('"');pa=[s.strip().strip('"') for s in d[1].split(',')] if len(d)>1 else []
  arr=np.full(tuple(len(states[p]) for p in pa)+(len(states[node]),),np.nan)
  table=re.search(r'\btable\s+([^;]+);',body,re.S)
  if table:
   if pa:raise ValueError('unsupported_parent_table_encoding')
   arr[:]=[float(s) for s in table[1].replace('\n',' ').split(',')]
  else:
   for config,vals in re.findall(r'\(([^)]+)\)\s*([^;]+);',body):
    idx=tuple(states[p].index(s.strip().strip('"')) for p,s in zip(pa,config.split(',')))
    arr[idx]=[float(s) for s in vals.split(',')]
  assert np.isfinite(arr).all() and np.allclose(arr.sum(-1),1)
  # BIF text can round row sums by a few parts in 1e8. Normalize the sampled proposal and retain strict saved-moment checks.
  assert np.max(abs(arr.sum(-1)-1))<1e-6
  arr=arr/arr.sum(-1,keepdims=True)
  specs[node]=(pa,arr)
 order=[]
 while len(order)<len(specs):order.append(min(n for n,(pa,_) in specs.items() if n not in order and set(pa)<=set(order)))
 return MSBN(order,[len(states[n]) for n in order],[tuple(order.index(p) for p in specs[n][0]) for n in order],[specs[n][1] for n in order])

def proposal(bn,evidence,target,state):
 ev,nodes=prune(bn,target,evidence);free=[j for j in sorted(nodes) if j not in ev]
 size=math.prod(bn.card[j] for j in free)
 if size>2**18:return None,size
 index=np.arange(size,dtype=np.int64);v={j:np.full(size,val,dtype=np.int16) for j,val in ev.items()};stride=1
 for j in reversed(free):v[j]=((index//stride)%bn.card[j]).astype(np.int16);stride*=bn.card[j]
 q=np.ones(size);w=np.ones(size)
 for j in sorted(nodes):
  probs=bn.cpts[j][tuple(v[p] for p in bn.parents[j])+(v[j],)]
  if j in ev:w*=probs
  else:q*=probs
 assert abs(q.sum()-1)<1e-8, ('proposal_mass',bn.names[0],size,float(q.sum()))
 I=(v[target]==state).astype(float);mu=q@w;p=q@(w*I)/mu
 var=q@(w*w*(I-p)**2)/mu**2
 unique,inverse=np.unique(np.column_stack([w,I]),axis=0,return_inverse=True)
 mass=np.bincount(inverse,weights=q,minlength=len(unique));keep=mass>0
 return (mass[keep]/mass[keep].sum(),unique[keep,0],unique[keep,1],p,var),size

def simulate(dist,N,reps,seed):
 q,w,I,*_=dist;rng=np.random.default_rng(seed);ys=[]
 for start in range(0,reps,16):
  counts=rng.multinomial(N,q,size=min(16,reps-start));den=counts@w;num=counts@(w*I)
  ys.extend(np.divide(num,den,out=np.full(len(den),np.nan),where=den>0).tolist())
 return np.array(ys)

def main():
 cases=json.loads((OUT/'private_cases.json').read_text());models={};instances=json.loads((ROOT/'artifacts/quantum_paper/benchmark_instances.json').read_text());results=[];elig=[]
 # Reconstruct only local evidence-pattern indices; no patient records are written.
 voc=json.loads((ROOT/'configs/vocab_v2.1.json').read_text())['propositions'];findings=[k for k,v in voc.items() if v['kind']!='diagnosis']
 df=pd.read_parquet(ROOT/'artifacts/v2.1/kb/cohort.parquet',columns=['subject_id',*findings]);subs=np.unique(df.subject_id)
 train=set(np.random.default_rng(2026).permutation(subs)[:int(round(.8*len(subs)))].tolist())
 values=df.loc[~df.subject_id.isin(train),findings].to_numpy(np.int8);patterns=np.unique(values,axis=0)
 for c in cases:
  cid=c['case_id']
  if c['population']=='public':
   if c['network'] not in models:models[c['network']]=public_bn(c['network'])
   bn=models[c['network']];r=instances[c['network']]['instances'][c['query_index']]
   ev={int(k):int(v) for k,v in r['evidence'].items()};target=c['target_index'];state=c['target_state']
  else:
   band=c['population']
   if band not in models:
    path=ROOT/('artifacts/v2.1/kb/bn.json' if band=='common' else 'artifacts/bands/uncommon/bn.json');m=json.loads(path.read_text())
    models[band]=MSBN(m['names'],[2]*len(m['names']),[tuple(p) for p in m['parents']],[np.array(a).reshape((2,)*(len(pa)+1)) for a,pa in zip(m['cpts'],m['parents'])])
   bn=models[band];ev={bn.names.index(k):int(v) for k,v in zip(findings,patterns[c['pattern']]) if v>=0};target=bn.names.index(c['target']);state=1
  dist,size=proposal(bn,ev,target,state)
  if dist is None:elig.append({'case_id':cid,'population':c['population'],'eligible':False,'reason':'state_space_cap','states':size});continue
  q,w,I,p,var=dist
  assert np.isclose(p,c['p'],rtol=1e-7,atol=1e-12),(cid,'posterior_mismatch',p,c['p'])
  expected=(1.645/.1)**2*var/p**2
  assert np.isclose(expected,c['lw_samples'],rtol=2e-6,atol=1e-5),(cid,'variance_mismatch',expected,c['lw_samples'])
  elig.append({'case_id':cid,'population':c['population'],'eligible':True,'states':size,'support':len(q),'posterior_rel_error':abs(p-c['p'])/p,'budget_rel_error':abs(expected-c['lw_samples'])/max(expected,1e-30)})
  for mode in ['relative','threshold']:
   delta=.1*(p if mode=='relative' else max(p,.01));N=max(1,int(math.ceil(1.645**2*var/delta**2)))
   if N>1e8:
    results.append({'case_id':cid,'population':c['population'],'cluster':c['cluster'],'mode':mode,'eligible':False,'reason':'sample_budget_cap','N':N});continue
   est=simulate(dist,N,256,970000000+cid*10000+int(mode=='threshold'))
   for rep,y in enumerate(est):results.append({'case_id':cid,'population':c['population'],'cluster':c['cluster'],'mode':mode,'eligible':True,'N':N,'rep':rep,'success':bool(np.isfinite(y) and abs(y-p)<=delta),'error_abs':float(abs(y-p)),'zero_weight':not bool(np.isfinite(y))})
  print(cid,c['population'],'proposal states',size,'compressed support',len(q),flush=True)
 pd.DataFrame(results).to_parquet(OUT/'classical_trials.parquet',index=False);(OUT/'classical_eligibility.json').write_text(json.dumps(elig,indent=2))
if __name__=='__main__':main()

