from pathlib import Path
import sys,json,math,hashlib
import numpy as np,pandas as pd
from scipy.optimize import linprog
O=Path(__file__).resolve().parent;S=O.parent;ROOT=S.parent

def envelope(a,p):
 r=np.clip(.5,a*p,1-a*(1-p))
 return p*(1-p)/(r*(1-r)),1/a

def run():
 rng=np.random.default_rng(2026111203);max_upper=0.;max_lower=0.
 for _ in range(20000):
  q=rng.dirichlet(np.ones(12));w=rng.beta(.4,.4,12);I=np.tile([0.,1.],6)
  a=q@w;p=(q@(w*I))/a;D=(q@(w*w*(I-p)**2))/(a*a*p*(1-p));lo,hi=envelope(a,p)
  max_upper=max(max_upper,D-hi);max_lower=max(max_lower,lo-D)
  assert lo-1e-10<=D<=hi+1e-10
 # Linear programs maximize the numerator over a fixed weight grid with prescribed first moments.
 lp=[];w=np.tile(np.linspace(0,1,101),2);I=np.repeat([0.,1.],101)
 for a in np.linspace(.05,.95,11):
  for p in np.linspace(.01,.99,11):
   cost=w*w*(I-p)**2/(a*a*p*(1-p));A=np.array([np.ones(len(w)),w,w*I]);b=[1,a,a*p]
   low=linprog(cost,A_eq=A,b_eq=b,bounds=(0,None),method='highs');high=linprog(-cost,A_eq=A,b_eq=b,bounds=(0,None),method='highs')
   assert low.success and high.success
   lower,upper=envelope(a,p);assert low.fun>=lower-1e-7 and abs(-high.fun-upper)<1e-7
   lp.append({'a_E':a,'p':p,'lower':lower,'grid_min':low.fun,'upper':upper,'grid_max':-high.fun})
 frames=[]
 for f in sorted((ROOT/'artifacts/quantum_paper').glob('bench_*.parquet')):
  d=pd.read_parquet(f);d=d[(d.status=='ok')&(d.p>0)&(d.p<1)].copy();d['D_check']=d.lw_samples*(.1/1.645)**2*d.p/(1-d.p);frames.append(d)
 d=pd.concat(frames,ignore_index=True);a=d.a_E.to_numpy();p=d.p.to_numpy();D=d.D_check.to_numpy();lo,hi=envelope(a,p)
 assert np.all(D>=lo-1e-7) and np.all(D<=hi+1e-7)
 idx=rng.integers(0,len(d),size=(3,100000));aa=a[idx[0]];pp=p[idx[1]];dd=D[idx[2]];ll,uu=envelope(aa,pp);invalid=(dd<ll-1e-7)|(dd>uu+1e-7)
 result={'protocol_sha256':hashlib.sha256((O/'AMENDMENT_003.md').read_bytes()).hexdigest(),'random_weighted_distributions':20000,'linear_programs_pairs':len(lp),'public_queries':len(d),'actual_query_violations':int(((D<lo-1e-7)|(D>hi+1e-7)).sum()),'independent_marginal_draws':100000,'infeasible_draws':int(invalid.sum()),'infeasible_fraction':float(invalid.mean()),'necessary_not_sufficient_for_BN_realizability':True,'a_times_D_range':[float(np.min(a*D)),float(np.max(a*D))]}
 (O/'feasibility_summary.json').write_text(json.dumps(result,indent=2));pd.DataFrame(lp).to_csv(O/'linear_program_checks.csv',index=False);print(json.dumps(result,indent=2))
if __name__=='__main__':run()
