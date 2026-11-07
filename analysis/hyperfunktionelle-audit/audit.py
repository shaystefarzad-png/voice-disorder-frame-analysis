"""Read-only audit. Existing Test is descriptive only; no tuning or new fits."""
import pathlib,openpyxl,numpy as np,json,collections,csv
root=pathlib.Path(__file__).resolve().parents[2];out=pathlib.Path(__file__).resolve().parent
book=openpyxl.load_workbook('/tmp/Hyperfunktionelle_Results.xlsx',read_only=True,data_only=True)
def table(sheet):
 rows=iter(book[sheet].values);h=next(rows);return [dict(zip(h,r)) for r in rows]
cohort={str(r['SubjectID']):r for r in table('SubjectSplit')};report={}
report['subjects']={f'{part}_{label}':sum(bool(r[part]) and r['Label']==label for r in cohort.values()) for part in ['Train','Test'] for label in [0,1]}
report['split_overlap']=sum(bool(r['Train']) and bool(r['Test']) for r in cohort.values())
report['summary']=table('Summary');report['cv_best_by_classifier']={}
for k in range(1,6):
 rows=table(f'CV_{k}');report['cv_best_by_classifier'][str(k)]={c:max((r for r in rows if r['Classifier']==c),key=lambda r:(r['MeanBA'],-r['StdBA'],r['Sensitivity'],-r['K'])) for c in ['SVM','KNN','RandomForest','NaiveBayes']}
report['voting_cv']=table('VotingSelection');report['nested_choices']=table('VotingNestedBranches')
subject_stats={};feature_rows=[];seen_groups={};report['workbooks']={}
for window in [25,75]:
 for group,label in [('Healthy',0),('Hyperfunktionelle',1)]:
  p=root/'datasets/corrected-v2'/f'FullFeats_{window}ms_{group}.xlsx';w=openpyxl.load_workbook(p,read_only=True,data_only=True);it=iter(w.active.values);h=next(it);ix={x:i for i,x in enumerate(h)}
  stats={};badlabel=0;duplicate=0;keys=set();nonfinite=np.zeros(43,dtype=int);trainfinite=np.zeros(43,dtype=int);trainmissing=np.zeros(43,dtype=int);vmin=np.full(43,np.inf);vmax=np.full(43,-np.inf);infs=0;nr=0
  for row in it:
   nr+=1;sid=str(row[ix['SubjectID']]);filename=str(row[ix['FileName']]);key=(sid,filename,row[ix['Vowel']],row[ix['FrameID']])
   duplicate+=key in keys;keys.add(key);badlabel+=row[ix['Label']]!=label or row[ix['Disease']]!=group
   seen_groups.setdefault(sid,set()).add(group)
   x=np.array([np.nan if v is None else v for v in row[12:]],dtype=float);finite=np.isfinite(x);nonfinite+=~finite;infs+=int(np.isinf(x).sum())
   st=stats.setdefault(sid,{'frames':0,'files':set(),'genders':set(),'active':0,'voiced':0});st['frames']+=1;st['files'].add(filename);st['genders'].add(str(row[ix['Gender']]));st['active']+=bool(row[ix['IsActive']]);st['voiced']+=bool(row[ix['IsVoiced']])
   if cohort[sid]['Train']:
    trainfinite+=finite;trainmissing+=~finite;vmin=np.minimum(vmin,np.where(finite,x,np.inf));vmax=np.maximum(vmax,np.where(finite,x,-np.inf))
  report['workbooks'][f'{window}_{group}']={'rows':nr,'subjects':len(stats),'files':sum(len(v['files']) for v in stats.values()),'bad_labels':badlabel,'duplicate_frame_keys':duplicate,'infinite_cells':infs,'nonfinite_fraction':float(nonfinite.sum()/(nr*43)),'training_all_missing_features':[h[12+j] for j in np.flatnonzero(trainfinite==0)],'training_constant_features':[h[12+j] for j in np.flatnonzero((vmin==vmax)&(trainfinite>0))]}
  for sid,st in stats.items():
   subject_stats[(window,sid)]={**st,'files':len(st['files']),'genders':','.join(sorted(st['genders'])),'label':label,'train':bool(cohort[sid]['Train']),'test':bool(cohort[sid]['Test'])}
  for j,name in enumerate(h[12:]):feature_rows.append({'window':window,'group':group,'feature':name,'missing_fraction':float(nonfinite[j]/nr),'training_missing_fraction':float(trainmissing[j]/max(1,trainmissing[j]+trainfinite[j]))})
report['cross_group_subject_overlap']=[sid for sid,g in seen_groups.items() if len(g)>1]
report['gender_and_quality']={}
for window in [25,75]:
 for part in ['train','test']:
  for label in [0,1]:
   a=[v for (win,sid),v in subject_stats.items() if win==window and v[part] and v['label']==label];frames=np.array([v['frames'] for v in a]);report['gender_and_quality'][f'{window}_{part}_{label}']={'subjects':len(a),'gender_counts':dict(collections.Counter(v['genders'] for v in a)),'frames_min_median_max':[int(frames.min()),float(np.median(frames)),int(frames.max())],'active_mean_subject':float(np.mean([v['active']/v['frames'] for v in a])),'voiced_mean_subject':float(np.mean([v['voiced']/v['frames'] for v in a])),'file_counts':dict(collections.Counter(v['files'] for v in a))}
report['nested_training_oof']={}
a=table('VotingTrainingOOF')
for mode in ['HardPrediction','SoftPrediction']:
 y=np.array([r['Label'] for r in a]);pred=np.array([bool(r[mode]) for r in a]);report['nested_training_oof'][mode]={'sensitivity':float(pred[y==1].mean()),'specificity':float((~pred[y==0]).mean()),'balanced_accuracy':float((pred[y==1].mean()+(~pred[y==0]).mean())/2),'FN':int(((y==1)&~pred).sum()),'FP':int(((y==0)&pred).sum())}
# Existing test scores are descriptive; no thresholds/models are selected here.
from sklearn.metrics import roc_auc_score
pred=table('SubjectPredictions');yy=np.array([v['Label'] for v in pred]);report['existing_test_score_diagnostics']={}
for col in ['S1_25ms_Probability','S2_75ms_Probability','S3A_Hybrid_Probability','Time25_Branch_Probability','Frequency75_Branch_Probability']:
    zz=np.array([v[col] for v in pred]);report['existing_test_score_diagnostics'][col]={'AUC':float(roc_auc_score(yy,zz)),'healthy_mean':float(zz[yy==0].mean()),'patient_mean':float(zz[yy==1].mean())}
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with (out/'feature_missingness.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=feature_rows[0].keys());writer.writeheader();writer.writerows(feature_rows)
print(json.dumps({k:v for k,v in report.items() if k not in ['summary','nested_choices']},ensure_ascii=False,indent=2))
