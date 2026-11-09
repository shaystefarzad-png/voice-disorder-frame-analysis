"""Inspect numeric checkpoint completion and benchmark timings; never fit a model."""
from pathlib import Path
from collections import Counter
import json,re,zipfile,xml.etree.ElementTree as ET
import h5py,numpy as np
ROOT=Path('approved-checkpoint-inspection');ROOT.mkdir(exist_ok=True)
CLASSES=['SVM_Linear','SVM_Quadratic','SVM_RBF','KNN3','KNN5','RandomForest','NaiveBayes']
RANKERS=['Ttest','CohenD','ReliefF','NCA']
def numbers(f,node):
 a=node[()]
 if h5py.check_dtype(ref=a.dtype):
  return np.concatenate([numbers(f,f[r]) for r in a.ravel() if r]).ravel()
 return np.asarray(a,dtype=float).ravel()
reports=[]
for disease in range(1,7):
 folder=Path('inspection-checkpoints')/f'approved-training-d{disease}'
 selected={}
 for p in folder.rglob('Fold_*.mat'):
  match=re.fullmatch(r'Fold_(\d+)(_partial)?\.mat',p.name)
  if match:
   key=(str(p.parent.relative_to(folder)),int(match[1]));prior=selected.get(key)
   if prior is None or not match[2]:selected[key]=p
 folds=[];counts=Counter()
 for (group,fold),p in sorted(selected.items()):
  try:
   with h5py.File(p) as f:ba=numbers(f,f['part']['BA'])
   ix=np.flatnonzero(np.isfinite(ba));cc=Counter(CLASSES[i%7] for i in ix);counts.update(cc)
   pending=np.flatnonzero(~np.isfinite(ba))
   nextcase=None
   if len(pending):
    i=int(pending[0]);nextcase={'Classifier':CLASSES[i%7],'Ranker':RANKERS[(i//7)%4],'K':i//28+1}
   folds.append({'Group':group,'Fold':fold,'File':p.name,'FiniteCandidateScores':len(ix),'GridCandidates':len(ba),'ByClassifier':dict(cc),'FirstUnfinishedCandidate':nextcase})
  except Exception as e:folds.append({'Group':group,'Fold':fold,'InspectionError':repr(e)})
 logs=[]
 for p in folder.rglob('*Log*.txt'):
  text=p.read_text(errors='replace');logs.append({'File':str(p.relative_to(folder)),'Tail':text[-5000:]})
 reports.append({'DiseaseIndex':disease,'CheckpointFiles':len(folds),'CompletedCandidateScores':sum(counts.values()),'ByClassifier':dict(counts),'Folds':folds,'Logs':logs,'FrozenSelectionFiles':len(list(folder.rglob('TRAINING_SELECTION_FROZEN.mat')))})
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
def spreadsheet(path):
 with zipfile.ZipFile(path) as z:
  shared=[]
  if 'xl/sharedStrings.xml' in z.namelist():shared=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(NS+'si')]
  rows=[]
  for row in ET.fromstring(z.read(next(n for n in z.namelist() if n.startswith('xl/worksheets/') and n.endswith('.xml') and '/_rels/' not in n))).iter(NS+'row'):
   values={}
   for c in row:
    ref=c.attrib['r'];letters=re.match('[A-Z]+',ref)[0];index=0
    for ch in letters:index=index*26+ord(ch)-64
    node=c.find(NS+'v');t=c.attrib.get('t')
    if t=='inlineStr':value=''.join(c.find(NS+'is').itertext())
    elif node is None:value=None
    elif t=='s':value=shared[int(node.text)]
    else:
     try:value=float(node.text)
     except ValueError:value=node.text
    values[index-1]=value
   rows.append([values.get(i) for i in range(max(values)+1)])
  return [dict(zip(rows[0],r)) for r in rows[1:]]
benchmarks=[]
for p in Path('inspection-benchmark').rglob('Training_Runtime_Benchmark.xlsx'):
 try:
  rows=spreadsheet(p)
  for r in rows:
   r.pop('TrainingValidationBA',None);r['Experiment']=str(p.parent)
  benchmarks.extend(rows)
 except Exception as e:benchmarks.append({'File':str(p),'Error':repr(e)})
request=json.loads(Path('ci/approved-checkpoint-inspect-request.json').read_text())
report={'SourceTrainingRun':request['training_run_id'],'SourceBenchmarkRun':request['benchmark_run_id'],'Diseases':reports,'BenchmarkTimings':benchmarks,'TestEvaluation':'NOT_PERFORMED'}
(ROOT/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
