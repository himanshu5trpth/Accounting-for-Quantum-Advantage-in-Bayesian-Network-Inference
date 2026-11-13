from pathlib import Path
import numpy as np,pandas as pd,json
from scipy import stats
from analyze import holm
O=Path(__file__).resolve().parent;s=json.loads((O/'summary.json').read_text());x=pd.read_csv(O/'query_summary.csv');rng=np.random.default_rng(2026111209)
for r in s['residual_diagnostics_exploratory']:
 pop,mode,policy=r['group'].split('|');g=x[(x.population==pop)&(x['mode']==mode)&(x.policy==policy)&(x.mean_calls>0)].copy()
 g['input']=np.log10(g.p if r['input']=='log_p' else g.a_E);g['residual']=np.log10(g.mean_calls/g.predicted);groups=[z for _,z in g.groupby('cluster')];vals=[]
 for _ in range(2000):
  z=pd.concat([groups[j].iloc[rng.integers(0,len(groups[j]),len(groups[j]))] for j in rng.integers(0,len(groups),len(groups))])
  val=stats.spearmanr(z['input'],z.residual).statistic
  if np.isfinite(val):vals.append(val)
 obs=r['rho'];r['iid_p_not_used_for_inference']=r.get('iid_p_not_used_for_inference',r.get('p'));r.pop('p',None);r['bootstrap_valid_replicates']=len(vals);r['bootstrap_undefined_replicates']=2000-len(vals);r['ci95_cluster']=np.quantile(vals,[.025,.975]).tolist();r['p_centered_cluster_bootstrap']=(1+sum(abs(v-obs)>=abs(obs) for v in vals))/(len(vals)+1)
for r,p in zip(s['residual_diagnostics_exploratory'],holm([r['p_centered_cluster_bootstrap'] for r in s['residual_diagnostics_exploratory']])):r['p_holm']=float(p)
s['rank_inference_note']='Exploratory residual-rank inference uses network/target-cluster bootstrap intervals and centered-bootstrap tests; independent-query p-values retained only as provenance.'
(O/'summary.json').write_text(json.dumps(s,indent=2));print('Updated 24 residual-association diagnostics with clustered uncertainty.')
