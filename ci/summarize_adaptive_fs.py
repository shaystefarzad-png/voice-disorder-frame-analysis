"""Progress for new adaptive+FS experiment, never imports preceding runs."""
from pathlib import Path
import json,os,hashlib
index=int(os.environ['VOICE_DISEASE_INDEX']);root=Path('adaptive-fs-real-new')/f'disease_{index}';root.mkdir(parents=True,exist_ok=True)
source=hashlib.sha256(Path('VoiceDiseaseModeling_AdaptiveFS.m').read_bytes()).hexdigest()
f=root/'CaseOutcome.json';outcome=json.loads(f.read_text()) if f.exists() else {}
folds=sum(1 for _ in root.rglob('Fold_*.mat'));windows=sum(1 for _ in root.rglob('Window_*.mat'))
complete=bool(outcome.get('Complete')) and bool(list(root.rglob('COMPLETE.mat'))) and (root/'Final_Report.json').exists()
can_resume=bool(outcome.get('PausedBudget')) or (not outcome and folds+windows>0)
report=dict(Experiment='Adaptive_Window_Then_CumulativeFS_v1',DiseaseIndex=index,RunID=os.environ['GITHUB_RUN_ID'],SourceSHA=source,Complete=complete,CanResume=can_resume,SavedFoldFiles=folds,SavedWindowPartitions=windows,Outcome=outcome)
(root/'Progress.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:f.write(f'\n### Adaptive + FS disease {index}\nComplete: {complete}; folds: {folds}; window partitions: {windows}. No prior experiment checkpoints used.\n')
