from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import beta,binomtest
O=Path(__file__).resolve().parent;S=O.parent
c=pd.concat([pd.read_parquet(p) for p in sorted((O/'classical_interval').glob('*.parquet'))],ignore_index=True);q=pd.concat([pd.read_parquet(p) for p in sorted((S/'followup').glob('*.parquet'))],ignore_index=True);cases={x['case_id']:x for x in json.loads((S/'private_cases.json').read_text())}
assert len(c)==34816 and not c.duplicated(['case_id','mode','rep']).any()
q=q[q.case_id.isin(c.case_id.unique())];assert len(q)==len(c)
rows=[]
for (cid,mode),z in c.groupby(['case_id','mode']):
 v=q[(q.case_id==cid)&(q['mode']==mode)];k=int(z.success.sum());n=len(z)
 rows.append({'case_id':int(cid),'population':cases[cid]['population'],'cluster':cases[cid]['cluster'],'mode':mode,'n':n,'classical_successes':k,'classical_complete':int(z.complete.sum()),'quantum_successes':int(v.success.sum()),'quantum_complete':int(v.complete.sum()),'classical_mean_calls':z.calls.mean(),'quantum_mean_calls':v.calls.mean(),'pass_ratio':v.calls.mean()/z.calls.mean(),'gate_adjusted_ratio':cases[cid]['gate_factor']*v.calls.mean()/z.calls.mean(),'p_under95':binomtest(k,n,.95,alternative='less').pvalue,'p_over95':binomtest(k,n,.95,alternative='greater').pvalue,'simultaneous95_lower':float(beta.ppf(.05/136,k,n-k+1)) if k else 0})
r=pd.DataFrame(rows)
for col in ['p_under95','p_over95']:
 order=np.argsort(r[col].to_numpy());adj=np.maximum.accumulate(r[col].to_numpy()[order]*(len(r)-np.arange(len(r))));out=np.empty(len(r));out[order]=np.minimum(1,adj);r[col+'_holm']=out
rng=np.random.default_rng(2026111204);groups={}
for key,g in r.groupby(['population','mode']):
 boot=[];clusters=g.cluster.unique()
 for _ in range(2000):
  bs=[]
  for cl in rng.choice(clusters,len(clusters),replace=True):
   gg=g[g.cluster==cl];bs.extend(rng.choice(gg.gate_adjusted_ratio.to_numpy(),len(gg),replace=True))
  boot.append(float(np.median(bs)))
 groups['|'.join(key)]={'queries':len(g),'clusters':len(clusters),'classical_successes':int(g.classical_successes.sum()),'quantum_successes':int(g.quantum_successes.sum()),'n':int(g.n.sum()),'classical_mean_calls_median':float(g.classical_mean_calls.median()),'quantum_mean_calls_median':float(g.quantum_mean_calls.median()),'median_pass_ratio':float(g.pass_ratio.median()),'median_gate_adjusted_ratio':float(g.gate_adjusted_ratio.median()),'gate_adjusted_ratio_query_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),'gate_adjusted_share_below1':float((g.gate_adjusted_ratio<1).mean())}
summary={'protocol_sha256':hashlib.sha256((O/'AMENDMENT_003.md').read_bytes()).hexdigest(),'new_classical_attempts':len(c),'new_classical_successes':int(c.success.sum()),'new_classical_completions':int(c.complete.sum()),'new_classical_interval_covers':int(c.interval_covers.sum()),'reused_quantum_attempts':len(q),'reused_quantum_successes':int(q.success.sum()),'classical_query_tolerance_cells':len(r),'classical_cells_under95_holm':int((r.p_under95_holm<.05).sum()),'classical_cells_over95_holm':int((r.p_over95_holm<.05).sum()),'min_simultaneous95_lower':float(r.simultaneous95_lower.min()),'public_networks':int(r.loc[r.population=='public','cluster'].nunique()),'matched_nominal_failure_probability':.05,'matched_achieved_coverage_or_optimal_cost':False,'cost_intervals':'cluster-and-query resampling of observed query means, conditional on observed within-query means; not a trial-level uncertainty interval','groups':groups}
r.to_csv(O/'matched_policy_query_summary.csv',index=False);(O/'matched_policy_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
