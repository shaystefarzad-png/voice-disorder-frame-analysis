"""Record progress of the frozen real-data protocol, without selecting models."""
import os,json,hashlib
from pathlib import Path
index=int(os.environ['VOICE_DISEASE_INDEX']);root=Path('final-complete-real')/f'disease_{index}';root.mkdir(parents=True,exist_ok=True)
source=hashlib.sha256(Path('VoiceDiseaseModeling_Final_Complete.m').read_bytes()).hexdigest()
expected='9500988d523447c198ffd4ee69e983b0404888b97a06424375ea1853eac5e184'
outfile=root/'CaseOutcome.json';outcome=json.loads(outfile.read_text()) if outfile.exists() else {}
complete=bool(outcome.get('Complete')) and bool(list(root.rglob('COMPLETE.mat'))) and (root/'Final_Report.json').exists()
paused=bool(outcome.get('PausedBudget'))
checkpoint_count=sum(1 for _ in root.rglob('Fold_*.mat'))
window_count=sum(1 for _ in root.rglob('Window_*.mat'))
# A step timeout may prevent CaseOutcome; only resume if checkpoints actually exist.
resumable=(paused or (not outcome and checkpoint_count+window_count>0)) and source==expected
report=dict(DiseaseIndex=index,RunID=os.environ['GITHUB_RUN_ID'],Attempt=int(os.environ['VOICE_COMPLETE_ATTEMPT']),
            SourceSHA=source,ValidatedSourceMatches=source==expected,Complete=complete,CanResume=bool(resumable),
            SavedFoldCheckpoints=checkpoint_count,CompletedWindowProbePartitions=window_count,Outcome=outcome,
            RepeatedSplitsPlannedForThisRun=1)
(root/'Progress.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
 f.write(f'\n### Disease {index}\nCompleted: **{complete}**. Fold checkpoint files: **{checkpoint_count}**. Completed window probes: **{window_count}**.\n')
