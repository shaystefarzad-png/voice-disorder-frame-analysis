"""Generate full, predeclared 10-fold grids; no screening by Test or pilot accuracy."""
import json,os
main=[];groups=[[],[],[]]
for d in range(1,7):
 for s in range(1,6):
  for b in range(1,6):main.append({'d':d,'scenario':s,'outer':0,'batch':b,'artifact':f'cv-main-d{d}-s{s}-b{b}'})
 for outer in range(1,11):
  g=0 if outer<=4 else (1 if outer<=8 else 2)
  for s in [4,5]:
   for b in range(1,6):groups[g].append({'d':d,'scenario':s,'outer':outer,'batch':b,'artifact':f'cv-inner-d{d}-o{outer}-s{s}-b{b}'})
alljobs=main+sum(groups,[])
assert [len(main)]+[len(x) for x in groups]==[150,240,240,120]
assert len({x['artifact'] for x in alljobs})==750
for job in alljobs:assert 1<=job['batch']<=5
with open(os.environ['GITHUB_OUTPUT'],'a') as f:
 for name,values in zip(['main','nested_a','nested_b','nested_c'],[main]+groups):f.write(name+'='+json.dumps({'include':values},separators=(',',':'))+'\n')
print('Full grid: 150 main CV jobs + 600 nested CV jobs, each with two folds; all features, all classifiers, all rankers.')
