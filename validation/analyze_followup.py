from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy import stats
from analyze import holm,cluster_boot,coverage_boot
O=Path(__file__).resolve().parent
c=pd.DataFrame(json.loads((O/'private_cases.json').read_text())).set_index('case_id')
d=pd.concat([pd.read_parquet(p) for p in sorted((O/'followup').glob('*.parquet'))],ignore_index=True)
x=d.groupby(['case_id','population','cluster','mode']).agg(n=('success','size'),successes=('success','sum'),rate=('success','mean'),completion=('complete','mean'),mean_calls=('calls','mean'),median_calls=('calls','median'),interval_coverage=('interval_covers','mean')).reset_index()
x['p']=x.case_id.map(c.p);x['a_E']=x.case_id.map(c.a_E);x['g']=x.case_id.map(c.gate_factor)
x['eps']=np.where(x['mode']=='relative',.1,.1*np.maximum(x.p,.01)/x.p)
x['lw']=np.maximum(1,x.case_id.map(c.lw_samples)*(.1/x.eps)**2)
x['reference']=math.pi/x.eps*(1/np.sqrt(x.a_E)+1/np.sqrt(x.a_E*x.p));x['kappa']=x.mean_calls/x.reference
x['Rg']=x.mean_calls*x.g/x.lw
x['p_greater_09']=[stats.binomtest(int(k),int(n),.9,alternative='greater').pvalue for k,n in zip(x.successes,x.n)]
x['p_holm']=holm(x.p_greater_09)
x['simultaneous_lower95_bonferroni']=[float(stats.beta.ppf(.05/len(x),k,n-k+1)) if k else 0 for k,n in zip(x.successes,x.n)]
x.to_csv(O/'followup_query_summary.csv',index=False)
s={'trials':len(d),'successes':int(d.success.sum()),'failures':int((~d.success).sum()),'completions':int(d.complete.sum()),'per_query_settings':len(x),'coverage_certified_holm':int((x.p_holm<.05).sum()),'minimum_simultaneous_lower95':float(x.simultaneous_lower95_bonferroni.min()),'all_intervals_cover_truth':int(d.interval_covers.sum()),'group_results':{}}
for keys,g in x.groupby(['population','mode']):
 s['group_results']['|'.join(keys)]={'queries':len(g),'trials':int(g.n.sum()),'success_rate':float(g.successes.sum()/g.n.sum()),'success_ci95_cluster':coverage_boot(g),'mean_completion':float(g.completion.mean()),'median_kappa':float(g.kappa.median()),'kappa_iqr':np.quantile(g.kappa,[.25,.75]).tolist(),'median_Rg':float(g.Rg.median()),'median_Rg_ci95_cluster':cluster_boot(g,lambda z:float(z.Rg.median())),'fraction_Rg_below_one':float((g.Rg<1).mean()),'coverage_certified_holm':int((g.p_holm<.05).sum()),'median_calls':float(g.mean_calls.median())}
(O/'followup_summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
