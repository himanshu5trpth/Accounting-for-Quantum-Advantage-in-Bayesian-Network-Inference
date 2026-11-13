"""Independent bounded-mean intervals and posterior stopping checks."""
import math
import numpy as np

def bernstein_interval(n,total,squares,alpha_one_sided):
 if n<2 or not 0<alpha_one_sided<1:raise ValueError('invalid sample count or failure allocation')
 mean=total/n;variance=max(0.,(squares-total*total/n)/(n-1))
 log=math.log(2/alpha_one_sided)
 radius=math.sqrt(2*variance*log/n)+7*log/(3*(n-1))
 return max(0.,mean-radius),min(1.,mean+radius)

def posterior_interval(e,t):
 if not all(math.isfinite(x) for x in [*e,*t]) or not (0<=e[0]<=e[1]<=1 and 0<=t[0]<=t[1]<=1):raise ValueError('invalid mean interval')
 lo=t[0]/e[1] if e[1]>0 else 0.;hi=min(1.,t[1]/e[0]) if e[0]>0 else 1.
 # Empty intersections are evidence of inconsistent inputs, never a stopping certificate.
 return (0.,1.) if lo>hi or lo>1 else (max(0.,lo),hi)

def stop(lo,hi,mode):
 if mode not in ('relative','threshold'):raise ValueError('unknown tolerance')
 return bool(math.isfinite(lo) and math.isfinite(hi) and 0<lo<=hi<=1 and (hi-lo)/2<=.1*(lo if mode=='relative' else max(lo,.01)))

def trial(dist,W,mode,seed):
 q,w,I,p,_=dist;u=w/W
 assert np.max(u)<=1+1e-10 and np.min(u)>=0
 u=np.minimum(u,1.);x=u*I;rng=np.random.default_rng(seed);counts=np.zeros(len(q),dtype=np.int64);nprev=0;bounds=(0.,1.)
 for epoch in range(24):
  n=128*2**epoch;counts+=rng.multinomial(n-nprev,q);nprev=n
  e=bernstein_interval(n,float(counts@u),float(counts@(u*u)),.05/(4*24))
  t=bernstein_interval(n,float(counts@x),float(counts@(x*x)),.05/(4*24))
  bounds=posterior_interval(e,t)
  if stop(*bounds,mode):return n,True,(bounds[0]+bounds[1])/2,bounds
 return nprev,False,(bounds[0]+bounds[1])/2,bounds
