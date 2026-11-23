"""Collect only newly completed adaptive+FS runs; incomplete cases stay explicit."""
from pathlib import Path
import json,math
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill
root=Path('adaptive-fs-progress');out=Path('adaptive-fs-collected');out.mkdir(exist_ok=True)
summary=[];stats=[];status=[]
for d in range(1,7):
 case=root/f'adaptive-fs-new-d{d}';p=json.loads((case/'Progress.json').read_text());status.append(p)
 if not p['Complete']:continue
 report=json.loads((case/'Final_Report.json').read_text());rows=report['Summary']
 assert len(rows)==4 and all(r['Design']=='D3_Adaptive' and r['Ranker'] not in ['None','None + None'] for r in rows)
 summary+=rows;stats+=report['PairedSubjectStatistics']
valid=sorted([r for r in stats if r.get('P_TwoSided') is not None and math.isfinite(r['P_TwoSided'])],key=lambda r:r['P_TwoSided']);previous=0
for i,r in enumerate(valid):previous=max(previous,(48-i)*r['P_TwoSided']);r['P_Holm']=min(1.,previous)
result=dict(Experiment='Adaptive_Window_Then_CumulativeFS_v1',CompletedDiseases=sum(s['Complete'] for s in status),PlannedDiseases=6,Summary=summary,PairedStatistics=stats,Status=status,PreviouslySeenTest=True)
(out/'AdaptiveFS_Report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
w=Workbook();w.remove(w.active)
for title,rows in [('Summary',summary),('PairedStatistics',stats),('Status',[dict(DiseaseIndex=s['DiseaseIndex'],Complete=s['Complete'],CanResume=s['CanResume'],SavedFoldFiles=s['SavedFoldFiles']) for s in status])]:
 s=w.create_sheet(title)
 if not rows:s.append(['No complete results yet']);continue
 keys=list(rows[0]);s.append(keys)
 for r in rows:s.append([None if isinstance(r.get(k),float) and not math.isfinite(r[k]) else r.get(k) for k in keys])
 s.freeze_panes='D2';s.auto_filter.ref=s.dimensions
 for c in s[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D')
 for j,k in enumerate(keys,1):
  from openpyxl.utils import get_column_letter
  s.column_dimensions[get_column_letter(j)].width=25
  if k.startswith('CV') or k.startswith('Test') and k not in ['TestSubjects','TestTrainRatio','TestTP','TestTN','TestFP','TestFN']:
   for col in s.iter_cols(min_col=j,max_col=j,min_row=2):
    for c in col:c.number_format='0.00%'
w.save(out/'AdaptiveFS_New_Results.xlsx')
print('New adaptive+FS completed diseases:',result['CompletedDiseases'])
