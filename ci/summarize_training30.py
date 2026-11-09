"""Count saved training candidates. Do not read or evaluate Test predictions."""
from pathlib import Path
import json,re,hashlib,os
from collections import Counter
import numpy as np,h5py
from scipy.io import loadmat
index=int(os.environ['VOICE_DISEASE_INDEX']);root=Path('approved-training30-results')/f'disease_{index}';root.mkdir(parents=True,exist_ok=True)
classes=['SVM_Linear','SVM_Quadratic','SVM_RBF','KNN3','KNN5','RandomForest','NaiveBayes']
def readba(p):
 if h5py.is_hdf5(p):
  with h5py.File(p) as f:
   value=f['part']['BA'][()]
   if h5py.check_dtype(ref=value.dtype):value=np.concatenate([f[r][()].ravel() for r in value.ravel()])
   return np.asarray(value,float).ravel()
 return np.asarray(loadmat(p,simplify_cells=True)['part']['BA'],float).ravel()
selected={};errors=[]
for p in root.rglob('Fold_*.mat'):
 m=re.fullmatch(r'Fold_(\d+)(_partial)?\.mat',p.name)
 if not m:continue
 key=(str(p.parent.relative_to(root)),int(m[1]))
 try:
  ba=readba(p);valid=np.flatnonzero(np.isfinite(ba));item={'Group':key[0],'Fold':key[1],'SavedCandidates':len(valid),'GridCandidates':len(ba),'File':p.name,'ByClassifier':dict(Counter(classes[i%7] for i in valid))}
  if key not in selected or item['SavedCandidates']>selected[key]['SavedCandidates']:selected[key]=item
 except Exception as e:errors.append({'File':str(p.relative_to(root)),'Error':repr(e)})
specfile=root/'Experiment_Specification.json';spec=json.loads(specfile.read_text()) if specfile.exists() else {}
modelsha=hashlib.sha256(Path('VoiceDiseaseModeling_Final.m').read_bytes()).hexdigest();manifestsha=hashlib.sha256(Path('datasets/approved-cycle43-v5/manifest.json').read_bytes()).hexdigest()
identity_ok=spec.get('DiseaseIndex')==index and spec.get('TrainingFrameCap')==30 and spec.get('NumericalSourceSHA')==modelsha and spec.get('InputManifestSHA')==manifestsha
failure=bool(list(root.rglob('LastFailure.mat')));test_models=bool(list(root.rglob('FinalModel_*.mat')))
ready=identity_ok and bool(list(root.rglob('TRAINING_SELECTION_FROZEN.mat'))) and not failure and not test_models
report={'DiseaseIndex':index,'RunID':os.environ['GITHUB_RUN_ID'],'Attempt':int(os.environ['VOICE_TRAIN30_ATTEMPT']),'TrainingFrameCap':30,'SourceIdentityVerified':identity_ok,'ModelSourceSHA':modelsha,'ReadyForFinalEvaluation':ready,'CanResume':identity_ok and not failure and not test_models and not errors,'NumericalFailureRecorded':failure,'TestModelsCreated':test_models,'SavedCandidateScores':sum(x['SavedCandidates'] for x in selected.values()),'CheckpointFolds':list(selected.values()),'InspectionErrors':errors}
(root/'Training_Progress.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write(f"\n### Disease {index}: training progress\nSaved candidate scores: **{report['SavedCandidateScores']}**. Selection frozen: **{ready}**. Test evaluated: **no**.\n")
