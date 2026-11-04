import os,pathlib,shutil,json,hashlib,zipfile
root=pathlib.Path.cwd();source=root/'final-results';out=root/'published-results';out.mkdir(exist_ok=True)
required=['All_Diseases_Summary.xlsx','Statistical_Hypothesis_Validation.xlsx']
assert all((source/x).is_file() for x in required),'Final summary/statistics are missing'
for p in source.rglob('*'):
 if p.is_file() and p.suffix.lower() in {'.xlsx','.png','.csv','.json','.txt'}:
  dest=out/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
prov=out/'provenance';prov.mkdir(exist_ok=True)
for name in ['VoiceDiseaseModeling_Final.m','runFinalStage.m','verifyLinearSVM.m','ci/finalStages.m.fragment','.github/workflows/full-research.yml','datasets/corrected-v2/Input_Manifest.json']:
 p=root/name;dest=prov/p.name;shutil.copy2(p,dest)
metadata={'run_id':os.environ['GITHUB_RUN_ID'],'source_commit':os.environ['GITHUB_SHA'],'diseases':6,'quick_test':False,'primary_cv_folds':10,'nested_voting_cv_folds':10,'all_training_frames':True,'svm':'linear hinge-loss SVM; equivalent validated scalable dual solver','test_after_all_training_frozen':True}
(prov/'Execution.json').write_text(json.dumps(metadata,indent=2))
manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'SHA256.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(out/'Final_Research_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='Final_Research_Results.zip':z.write(p,str(p.relative_to(out)))
print('Final research reports packaged. Model checkpoints remain in stage artifacts for 30 days.')
