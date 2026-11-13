from pathlib import Path
import json
import pandas as pd
from scipy import stats
from analyze import holm
from study import exact_interval
O=Path(__file__).resolve().parent
x=pd.read_csv(O/'classical_query_summary.csv');ci=[exact_interval(int(k),int(n)) for k,n in zip(x.successes,x.n)];x['ci_low']=[a for a,b in ci];x['ci_high']=[b for a,b in ci]
x['p_undercoverage']=[stats.binomtest(int(k),int(n),.9,alternative='less').pvalue for k,n in zip(x.successes,x.n)];x['p_undercoverage_holm']=holm(x.p_undercoverage);x.to_csv(O/'classical_query_summary.csv',index=False)
s=json.loads((O/'summary.json').read_text());s['classical_binomial_family_size']=len(x);s['classical_undercoverage_holm_cases']=int((x.p_undercoverage_holm<.05).sum());(O/'summary.json').write_text(json.dumps(s,indent=2));print('Classical per-query undercoverage after Holm',s['classical_undercoverage_holm_cases'],'of',len(x))
