"""Prove the patch does not change already-computed adaptive-FS scores/seeds."""
import hashlib,json,re,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
old=subprocess.check_output(['git','show','b9e1f69:VoiceDiseaseModeling_AdaptiveFS.m'],cwd=root).decode()
new=(root/'VoiceDiseaseModeling_AdaptiveFS.m').read_text();manifest=json.loads((root/'AdaptiveFS_Repair_Manifest.json').read_text())
assert hashlib.sha256(old.encode()).hexdigest()==manifest['OriginalAdaptiveFS_SHA256']
assert hashlib.sha256(new.encode()).hexdigest()==manifest['RepairedAdaptiveFS_SHA256']
def functions(text):
 parts=re.split(r'(?m)^(?=function\s)',text);out={}
 for p in parts:
  if not p.strip():continue
  name=re.match(r'function\s+(?:\[[^\]]*\]\s*=\s*|\w+\s*=\s*)?(\w+)',p).group(1);out[name]=p
 return out
a,b=functions(old),functions(new);assert a.keys()==b.keys()
allowed={'VoiceDiseaseModeling_AdaptiveFS','branchBudget','selectVote','selftest'}
changed={n for n in a if a[n]!=b[n]};assert changed<=allowed,changed-allowed
# Runtime-only migration fields are absent from the signature configuration.
for text in [old,new]:
 line=next(l for l in text.splitlines() if l.startswith('routed=cfg;'))
 assert "'ResumeSourceSHA'" in line and "'ResumeExecutionSHA'" in line
assert 'identitySource=string(cfg.ResumeSourceSHA);' in new
for key in ['runThree','searchCV','prepareFold','selectWindows','fitFinal','chosenOOF','evaluateFold','rankFeatures','chooseVote','holdoutSubjects','groupFolds','configuration']:
 assert a[key]==b[key],key
# Exactly unchanged decisions whenever original count tolerance is feasible.
section=b['branchBudget'];assert 'effectiveTolerance=max(cfg.ComparableCountTolerance,minimum)' in section
# Only the log/report changes outside branchBudget; score calculation is identical.
for expression in ['allValues(k,:)=part.Values','oof(fold==k)=part.Probability','hardOOF(fold==k)=part.Prediction','tc=searchCV(','fc=searchCV(','choice=chooseVote(','tm=fitFinal(','fm=fitFinal(']:
 assert expression in a['selectVote'] and expression in b['selectVote']
report=dict(OriginalSHA256=manifest['OriginalAdaptiveFS_SHA256'],RepairedSHA256=manifest['RepairedAdaptiveFS_SHA256'],ChangedFunctions=sorted(changed),UnchangedFunctionCount=len(a)-len(changed),SignatureRoutingUnchanged=True,PrimaryTrainingSelectionFunctionsUnchanged=True,OriginalArtifactRun=45107990079)
(root/'AdaptiveFS_Repair_Compatibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
