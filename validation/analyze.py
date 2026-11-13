from pathlib import Path
import json,math,itertools
import numpy as np,pandas as pd
from scipy import stats
from study import exact_interval
O=Path(__file__).resolve().parent
rng=np.random.default_rng(2026111203);B=2000

def holm(pvals):
 p=np.asarray(pvals,float);idx=np.argsort(p);out=np.empty(len(p));out[idx]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[idx]));return out

def cluster_boot(g,fn):
 groups=[v for _,v in g.groupby('cluster')];ans=[]
 for _ in range(B):
  sample=pd.concat([groups[i].iloc[rng.integers(0,len(groups[i]),len(groups[i]))] for i in rng.integers(0,len(groups),len(groups))],ignore_index=True)
  ans.append(fn(sample))
 return np.quantile(ans,[.025,.975]).tolist()

def coverage_boot(g):
 groups=[v for _,v in g.groupby('cluster')];out=[]
 for _ in range(B):
  picked=[groups[i] for i in rng.integers(0,len(groups),len(groups))];counts=[]
  for x in picked:
   z=x.iloc[rng.integers(0,len(x),len(x))]
   counts.extend(rng.binomial(z.n.to_numpy(int),z.rate.to_numpy())/z.n.to_numpy())
  out.append(np.mean(counts))
 return np.quantile(out,[.025,.975]).tolist()

def main():
 cases=pd.DataFrame(json.loads((O/'private_cases.json').read_text())).set_index('case_id')
 d=pd.concat([pd.read_parquet(p) for p in sorted((O/'results').glob('*.parquet'))],ignore_index=True)
 cell=d.groupby(['case_id','population','cluster','mode','policy']).agg(n=('success','size'),successes=('success','sum'),rate=('success','mean'),completion=('complete','mean'),mean_calls=('calls','mean'),median_calls=('calls','median'),mean_legacy=('legacy_calls','mean'),interval_coverage=('interval_covers','mean')).reset_index()
 cell['p']=cell.case_id.map(cases.p);cell['a_E']=cell.case_id.map(cases.a_E)
 cell['lw_base']=cell.case_id.map(cases.lw_samples);cell['gate']=cell.case_id.map(cases.gate_factor)
 cell['relative_tolerance']=np.where(cell['mode']=='relative',.1,.1*np.maximum(cell.p,.01)/cell.p)
 cell['lw_budget']=np.maximum(1,cell.lw_base*(.1/cell.relative_tolerance)**2)
 cell['reference']=math.pi/cell.relative_tolerance*(1/np.sqrt(cell.a_E)+1/np.sqrt(cell.a_E*cell.p))
 cell['kappa']=cell.mean_calls/cell.reference
 cell['historical_kappa']=np.where(cell.population=='public',23.9928831123963,11.685850819903646)
 cell['predicted']=cell.historical_kappa*cell.reference
 cell['Rg']=cell.mean_calls*cell.gate/cell.lw_budget
 cell['accuracy_qualified_advantage']=(cell.rate>=.9)&(cell.Rg<1)
 cell['p_coverage_greater_09']=[stats.binomtest(int(k),int(n),.9,alternative='greater').pvalue for k,n in zip(cell.successes,cell.n)]
 cell['p_undercoverage']=[stats.binomtest(int(k),int(n),.9,alternative='less').pvalue for k,n in zip(cell.successes,cell.n)]
 cell['p_coverage_holm']=holm(cell.p_coverage_greater_09);cell['p_undercoverage_holm']=holm(cell.p_undercoverage)
 cis=[exact_interval(int(k),int(n)) for k,n in zip(cell.successes,cell.n)]
 cell['ci_low']=[c[0] for c in cis];cell['ci_high']=[c[1] for c in cis]
 cell.to_csv(O/'query_summary.csv',index=False)
 groups={};cal=[];residual=[];precision=[]
 for keys,g in cell.groupby(['population','mode','policy']):
  pop,mode,policy=keys;key='|'.join(keys);valid=g[g.mean_calls>0];err=np.log10(valid.mean_calls/valid.predicted)
  x=np.log10(valid.predicted.to_numpy());y=np.log10(valid.mean_calls.to_numpy());slope,intercept=np.polyfit(x,y,1) if np.ptp(x)>1e-9 else (float('nan'),float('nan'))
  slope_ci=cluster_boot(valid,lambda z:float(np.polyfit(np.log10(z.predicted),np.log10(z.mean_calls),1)[0])) if len(valid)>2 else []
  groups[key]={'queries':len(g),'clusters':int(g.cluster.nunique()),'trials':int(g.n.sum()),'success_rate':float(g.successes.sum()/g.n.sum()),'success_ci95_two_level_cluster':coverage_boot(g),'completion_rate':float(g.completion.mean()),'queries_coverage_certified_holm':int((g.p_coverage_holm<.05).sum()),'queries_undercoverage_holm':int((g.p_undercoverage_holm<.05).sum()),'queries_rate_at_least_09':int((g.rate>=.9).sum()),'median_mean_calls':float(g.mean_calls.median()),'median_kappa_among_attempted':float(valid.kappa.median()),'kappa_iqr_among_attempted':np.quantile(valid.kappa,[.25,.75]).tolist(),'zero_work_training_failed_queries':int((g.mean_calls==0).sum()),'median_Rg_attempted':float(valid.Rg.median()),'accuracy_qualified_advantage_fraction':float(g.accuracy_qualified_advantage.mean()),'legacy_count_median_relative_excess':float(np.median(valid.mean_legacy/valid.mean_calls-1)),'median_work_over_historical_prediction':float(np.median(valid.mean_calls/valid.predicted)),'median_work_over_historical_prediction_ci95':cluster_boot(valid,lambda z:float(np.median(z.mean_calls/z.predicted))),'log10_cost_slope':float(slope),'log10_cost_intercept':float(intercept),'log10_cost_slope_ci95_cluster':slope_ci,'median_absolute_log10_prediction_error':float(np.median(abs(err))),'ci95_median_Rg':cluster_boot(valid,lambda z:float(z.Rg.median())),'leave_one_cluster_out_success_range':[float(g[g.cluster!=k].rate.mean()) for k in sorted(g.cluster.unique())]}
  cal.append({'group':key,'slope':slope,'slope_ci95':slope_ci,'intercept':intercept})
  for label,z in [('log_p',np.log10(valid.p)),('log_a_E',np.log10(valid.a_E))]:
   r=stats.spearmanr(z,np.log10(valid.mean_calls/valid.predicted));residual.append({'group':key,'input':label,'rho':float(r.statistic),'p':float(r.pvalue)})
 # Paired, per-case McNemar tests, keeping discordant failures.
 pairs=d.pivot(index=['case_id','population','cluster','mode','rep'],columns='policy',values='success').reset_index();mcn=[]
 for keys,g in pairs.groupby(['case_id','population','cluster','mode']):
  a=int((g.historical&~g.interval).sum());b=int((~g.historical&g.interval).sum())
  mcn.append(dict(zip(['case_id','population','cluster','mode'],keys))|{'historical_only':a,'interval_only':b,'p':float(stats.binomtest(a,a+b,.5).pvalue) if a+b else 1.,'difference':float(g.interval.mean()-g.historical.mean())})
 mc=pd.DataFrame(mcn);mc['p_holm']=holm(mc.p);mc.to_csv(O/'paired_mcnemar.csv',index=False)
 paired={}
 for keys,g in mc.groupby(['population','mode']):paired['|'.join(keys)]={'mean_success_difference_interval_minus_historical':float(g.difference.mean()),'ci95_cluster':cluster_boot(g,lambda z:float(z.difference.mean())),'significant_cases_holm':int((g.p_holm<.05).sum())}
 # Cost changes with the precision convention, tested on cluster-average paired log ratios.
 for keys,g in cell.groupby(['population','policy']):
  z=g.pivot(index=['case_id','cluster'],columns='mode',values='mean_calls').dropna();z=z[(z.relative>0)&(z.threshold>0)].reset_index();z['log_ratio']=np.log10(z.threshold/z.relative)
  cl=z.groupby('cluster').log_ratio.mean().to_numpy();obs=abs(cl.mean())
  if len(cl)<=14:
   vals=np.array([abs(np.mean(cl*np.array(sign))) for sign in itertools.product([-1,1],repeat=len(cl))]);p=float((vals>=obs-1e-15).mean())
  else:p=float('nan')
  precision.append({'group':'|'.join(keys),'median_threshold_to_relative_cost':float(np.median(10**z.log_ratio)),'ci95_cluster':cluster_boot(z,lambda x:float(np.median(10**x.log_ratio))),'cluster_signflip_p':p})
 for r,p in zip(precision,holm([r['cluster_signflip_p'] for r in precision])):r['p_holm']=float(p)
 for r,p in zip(residual,holm([r['p'] if np.isfinite(r['p']) else 1 for r in residual])):r['p_holm']=float(p)
 # Leave-one-network-out overhead prediction on public cases, conditional on actual attempts.
 loo=[]
 for mode in ['relative','threshold']:
  for policy in ['historical','interval']:
   g=cell[(cell.population=='public')&(cell['mode']==mode)&(cell.policy==policy)&(cell.mean_calls>0)]
   for net in sorted(g.cluster.unique()):
    train=g[g.cluster!=net];test=g[g.cluster==net];k=float(train.kappa.median());loo.append({'mode':mode,'policy':policy,'network':net,'training_kappa':k,'median_abs_log10_error':float(np.median(abs(np.log10(test.mean_calls/(k*test.reference))))),'test_queries':len(test)})
 result={'protocol_sha256':json.loads((O/'selection_manifest.json').read_text())['protocol_sha256'],'queries':len(cases),'quantum_posterior_trials':len(d),'repeats_per_query_setting':256,'coverage_family_size':len(cell),'paired_family_size':len(mc),'groups':groups,'paired_policy_comparison':paired,'precision_comparison':precision,'calibration':cal,'residual_diagnostics_exploratory':residual,'leave_one_network_out':loo,'failure_counts':d.groupby(['policy','error']).size().reset_index(name='n').to_dict('records'),'all_failures_retained':True}
 # Classical results: finite-sample estimator outcomes, not normal draws.
 e=pd.read_parquet(O/'classical_trials.parquet');e=e[e.eligible==True]
 ccell=e.groupby(['case_id','population','cluster','mode']).agg(n=('success','size'),successes=('success','sum'),rate=('success','mean'),N=('N','first')).reset_index();cs={}
 for key,g in ccell.groupby(['population','mode']):cs['|'.join(key)]={'queries':len(g),'trials':int(g.n.sum()),'success_rate':float(g.successes.sum()/g.n.sum()),'success_ci95_cluster':coverage_boot(g),'min_case_success':float(g.rate.min()),'max_case_success':float(g.rate.max()),'median_N':float(g.N.median()),'cases_below_09':int((g.rate<.9).sum())}
 result['classical']=cs;result['classical_trials']=len(e);result['classical_eligible_queries']=int(ccell.case_id.nunique());result['classical_exclusions']=pd.DataFrame(json.loads((O/'classical_eligibility.json').read_text())).groupby(['population','eligible']).size().reset_index(name='n').to_dict('records')
 ccell.to_csv(O/'classical_query_summary.csv',index=False)
 (O/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=True));print(json.dumps({'quantum_posterior_trials':len(d),'classical_trials':len(e),'classical':cs},indent=2),flush=True)
if __name__=='__main__':main()
