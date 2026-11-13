from pathlib import Path
import sys,json,math,concurrent.futures as cf,hashlib
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;S=O.parent;ROOT=S.parent;sys.path.insert(0,str(S));sys.path.insert(0,str(ROOT/'scripts'))
import followup as F,study as B
from classical import MSBN,proposal
from msbn import prune,node_max
from confidence import trial

def select():
 parts=[]
 for band in ['common','uncommon']:
  P=ROOT/'artifacts/bands'/band
  d=pd.concat([pd.read_parquet(p,columns=['pattern','target','p']) for p in sorted(P.glob('c3_patterns_*.parquet'))]).drop_duplicates(['pattern','target'])
  w=pd.concat([pd.read_parquet(p,columns=['pattern','target','lw_samples','p_evidence','logW','ratio_v1']) for p in sorted(P.glob('w6_*.parquet'))]).drop_duplicates(['pattern','target'])
  adm=pd.read_parquet(P/'c3_admissions.parquet',columns=['pattern']).value_counts().rename('weight').reset_index();d=d.merge(w,on=['pattern','target']).merge(adm,on='pattern');h=json.loads((P/'c2.json').read_text());drop={t for t,v in h['auroc'].items() if v['distinct_posteriors']==1};d=d[(d.p>=.001)&(d.p<=.04)&(~d.target.isin(drop))].copy();d['population']=band;d['a_E']=d.p_evidence/np.exp(d.logW);parts.append(d)
 d=pd.concat(parts,ignore_index=True).sort_values(['population','target','pattern']).reset_index(drop=True);assert d.weight.sum()==733226
 rng=np.random.default_rng(2026111205);idx=rng.choice(len(d),256,replace=True,p=d.weight/d.weight.sum());chosen=d.iloc[idx].copy().reset_index(drop=True);chosen['draw_id']=np.arange(256);chosen.to_json(O/'private_population_cases.json',orient='records',indent=2)
 old={(c['population'],c.get('target'),c.get('pattern')) for c in json.loads((S/'private_cases.json').read_text()) if c['population']!='public'}
 manifest={'population_query_positions':int(d.weight.sum()),'sampling':'iid with replacement, admission-multiplicity weighted','draw_positions':256,'distinct_pattern_target_configurations':int(chosen.drop_duplicates(['population','target','pattern']).shape[0]),'draw_positions_overlapping_earlier_validation':sum((r.population,r.target,r.pattern) in old for r in chosen.itertuples()),'protocol_sha256':hashlib.sha256((O/'AMENDMENT_004.md').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((O/'private_population_cases.json').read_bytes()).hexdigest(),'selection_seed':2026111205}
 (O/'population_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest,flush=True)

def job(args):
 c,dist,W=args;rows=[];p=c['p'];N=max(1.,c['lw_samples']*min(1.,p/.01)**2)
 for rep in range(64):
  seed=int(np.random.SeedSequence([1210000000,c['draw_id'],rep]).generate_state(1,dtype=np.uint64)[0]);B.TRIAL_CALLS=0
  try:
   z=F.robust(c['a_E'],p,'threshold',seed);rows.append({'draw_id':c['draw_id'],'population':c['population'],'rep':rep,'policy':'quantum100','calls':B.TRIAL_CALLS,'complete':z['complete'],'success':bool(z['complete'] and abs(z['estimate']-p)<=.1*max(p,.01)),'ratio_asymptotic':B.TRIAL_CALLS*c['ratio_v1']/N})
  except Exception as err:rows.append({'draw_id':c['draw_id'],'population':c['population'],'rep':rep,'policy':'quantum100','calls':B.TRIAL_CALLS,'complete':False,'success':False,'error':type(err).__name__,'ratio_asymptotic':B.TRIAL_CALLS*c['ratio_v1']/N})
  if dist is not None:
   seed=int(np.random.SeedSequence([1310000000,c['draw_id'],rep]).generate_state(1,dtype=np.uint64)[0]);n,done,y,ci=trial(dist,W,'threshold',seed);rows.append({'draw_id':c['draw_id'],'population':c['population'],'rep':rep,'policy':'classicalEB','calls':n,'complete':done,'success':bool(done and abs(y-p)<=.1*max(p,.01))})
 pd.DataFrame(rows).to_parquet(O/'population_trials'/f"{c['draw_id']:03d}.parquet",index=False)
 return c['draw_id'],dist is not None

def run():
 (O/'population_trials').mkdir(exist_ok=True);cases=json.loads((O/'private_population_cases.json').read_text());models={}
 voc=json.loads((ROOT/'configs/vocab_v2.1.json').read_text())['propositions'];findings=[k for k,v in voc.items() if v['kind']!='diagnosis'];df=pd.read_parquet(ROOT/'artifacts/v2.1/kb/cohort.parquet',columns=['subject_id',*findings]);subs=np.unique(df.subject_id);train=set(np.random.default_rng(2026).permutation(subs)[:int(round(.8*len(subs)))].tolist());patterns=np.unique(df.loc[~df.subject_id.isin(train),findings].to_numpy(np.int8),axis=0);tasks=[]
 for c in cases:
  band=c['population']
  if band not in models:
   m=json.loads((ROOT/('artifacts/v2.1/kb/bn.json' if band=='common' else 'artifacts/bands/uncommon/bn.json')).read_text());models[band]=MSBN(m['names'],[2]*len(m['names']),[tuple(p) for p in m['parents']],[np.array(a).reshape((2,)*(len(pa)+1)) for a,pa in zip(m['cpts'],m['parents'])])
  bn=models[band];ev={bn.names.index(k):int(v) for k,v in zip(findings,patterns[c['pattern']]) if v>=0};target=bn.names.index(c['target']);dist,size=proposal(bn,ev,target,1);pev,_=prune(bn,target,ev);W=math.prod(node_max(bn,e,v,pev) for e,v in pev.items())
  if dist is not None:assert np.isclose(dist[0]@dist[1]/W,c['a_E'],rtol=1e-7) and np.isclose(dist[3],c['p'],rtol=1e-7)
  if not (O/'population_trials'/f"{c['draw_id']:03d}.parquet").exists():tasks.append((c,dist,W))
 with cf.ProcessPoolExecutor(max_workers=4) as pool:
  for i,r in enumerate(pool.map(job,tasks)):
   if i%16==0 or i==len(tasks)-1:print(i+1,r,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='select':select()
 else:run()
