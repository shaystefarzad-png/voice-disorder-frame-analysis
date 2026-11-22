"""Read saved completed evaluations. Never fit or select models or touch Test anew."""
import json, math
from pathlib import Path
from openpyxl import load_workbook, Workbook
ROOT=Path('downloaded-final-complete');OUT=Path('final-complete-collected-results');OUT.mkdir(exist_ok=True)
DISEASES=['Rekurrensparese','Hyperfunktionelle','Laryngitis','Reinke Ödem','Spasmodische Dysphonie','Stimmlippenpolyp']
def sheet(file,name):
    wb=load_workbook(file,read_only=True,data_only=True)
    if name not in wb.sheetnames:wb.close();return []
    it=iter(wb[name].values);header=next(it,());data=[dict(zip(header,row)) for row in it if any(v is not None for v in row)];wb.close();return data
summary=[];paired=[];status=[]
for i,disease in enumerate(DISEASES,1):
    directory=ROOT/f'final-complete-real-d{i}'
    complete=list(directory.rglob('COMPLETE.mat'));books=list(directory.rglob('All_Diseases_Summary.xlsx'))
    outcome_file=directory/'CaseOutcome.json';outcome=json.loads(outcome_file.read_text()) if outcome_file.exists() else {}
    ready=len(complete)==1 and len(books)==1
    rows=sheet(books[0],'Summary') if ready else []
    ready=ready and len(rows)==12 and all(r['Disease']==disease and r['Repeat']==1 for r in rows)
    entry=dict(Disease=disease,Complete=bool(ready),SavedCompleteCases=len(complete),RunDeclaredComplete=outcome.get('Complete',False))
    status.append(entry)
    if not ready:continue
    for row in rows:
        tp,tn,fp,fn=(row['Test'+k] for k in ['TP','TN','FP','FN'])
        assert tp+fn>0 and tn+fp>0
        assert abs(row['TestAccuracy']-(tp+tn)/(tp+tn+fp+fn))<1e-10
        assert abs(row['TestSensitivity']-tp/(tp+fn))<1e-10
        assert abs(row['TestBalancedAccuracy']-(tp/(tp+fn)+tn/(tn+fp))/2)<1e-10
    summary+=rows
    stats=list(directory.rglob('Statistical_Hypothesis_Validation.xlsx'));assert len(stats)==1
    items=sheet(stats[0],'PairedSubjectTests');assert len(items)==24
    paired+=items
assert summary,'No completed real evaluations are available.'
# Both metrics, three designs, two hybrids, two single windows, six planned diseases.
planned_family=2*3*2*2*len(DISEASES)
for repeat in sorted({r['Repeat'] for r in paired}):
    group=[r for r in paired if r['Repeat']==repeat and isinstance(r.get('P_TwoSided'),(int,float)) and math.isfinite(r['P_TwoSided'])]
    previous=0.
    for rank,row in enumerate(sorted(group,key=lambda x:x['P_TwoSided'])):
        previous=max(previous,(planned_family-rank)*row['P_TwoSided']);row['P_Holm']=min(1.,previous)
for row in paired:
    row['ExploratoryEvidence']=row['Delta']>0 and row['CI95Lower']>0 and row['P_Holm']<.05
    row['IndependentConfirmation']=False
    model=next(r for r in summary if r['Disease']==row['Disease'] and r['Design']==row['Design'] and r['Scenario']==row['Hybrid'])
    row['BothFeatureGroupsPresent']=bool(model.get('ContainsBothFeatureGroups',False))
    if row['Hybrid']=='S3B_Voting':row['BothFeatureGroupsPresent']=bool(model.get('GenuineVoting',False))
verdicts=[]
for state in status:
 for design in ['D1_FS','D2_NoFS','D3_Adaptive']:
    entry=dict(Disease=state['Disease'],Design=design,Complete=state['Complete'],IndependentHypothesisConfirmed=False)
    for hybrid in ['S3A_Hybrid','S3B_Voting']:
        rows=[r for r in paired if r['Disease']==state['Disease'] and r['Design']==design and r['Hybrid']==hybrid and r['Metric']=='BalancedAccuracy']
        entry[hybrid+'_BAEvidenceAgainstBothWindows']=len(rows)==2 and all(r['ExploratoryEvidence'] and r['BothFeatureGroupsPresent'] for r in rows)
        entry[hybrid+'_BAEvidenceAgainstEitherWindow']=any(r['ExploratoryEvidence'] and r['BothFeatureGroupsPresent'] for r in rows)
    verdicts.append(entry)
report=dict(SourceRun=44480674288,TrainingPerformedByCollector=False,CompletedDiseases=sum(r['Complete'] for r in status),PlannedDiseases=6,RepeatedSplits=1,
            Role='Exploratory after prior Test inspection; incomplete disease excluded; no independent confirmation',PlannedHolmFamily=planned_family,
            Status=status,Summary=summary,PairedStatistics=paired,Verdicts=verdicts)
(OUT/'Collected_Report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
wb=Workbook();wb.remove(wb.active)
for name,rows in [('Summary',summary),('PairedStatistics',paired),('HypothesisStatus',verdicts),('CompletionStatus',status)]:
 ws=wb.create_sheet(name)
 if rows:
  fields=list(rows[0]);ws.append(fields)
  for row in rows:ws.append([row.get(k) for k in fields])
  ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
ws=wb.create_sheet('Interpretation');ws.append(['No retraining or model selection was performed.']);ws.append(['Only completed frozen evaluations are included.']);ws.append(['One repeat, previously inspected Test: no independent confirmation.']);ws.append(['Holm recalculated across both metrics and all six planned diseases (144 comparisons), with missing disease retained in family size.'])
wb.save(OUT/'Collected_Real_Results.xlsx')
print(json.dumps(dict(CompletedDiseases=report['CompletedDiseases'],Rows=len(summary),Status=status),ensure_ascii=False,indent=2))
