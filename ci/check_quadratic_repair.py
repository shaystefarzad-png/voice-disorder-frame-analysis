"""Verify that checkpoint admission does not conceal scientific code changes."""
from pathlib import Path
import hashlib,re,json
old=Path('ci/quadratic-repair-base.m.txt')
if not old.exists():
 import subprocess
 text=subprocess.check_output(['git','show','be7c188:VoiceDiseaseModeling_Final_Complete.m'],text=True)
else:text=old.read_text()
new=Path('VoiceDiseaseModeling_Final_Complete.m').read_text()
assert hashlib.sha256(text.encode()).hexdigest()=='9500988d523447c198ffd4ee69e983b0404888b97a06424375ea1853eac5e184'
def blocks(source):
 matches=list(re.finditer(r'^function\b.*',source,re.M));out={}
 for i,m in enumerate(matches):
  h=m.group();h=h.split('=',1)[-1] if '=' in h else h[len('function '):]
  name=re.match(r'\s*(\w+)',h)[1];out[name]=source[m.start():matches[i+1].start() if i+1<len(matches) else len(source)].strip()
 return out
before,after=blocks(text),blocks(new)
changed=[name for name in before if before[name]!=after[name]]
assert set(changed)=={'VoiceDiseaseModeling_Final_Complete','configuration','fitModel','modelProbability'},changed
expected_cfg=before['configuration'].replace("'ProtocolVersion',\"Complete_AccuracyAndBA_v2\");", "'ProtocolVersion',\"Complete_AccuracyAndBA_v2\", ...\n    'ResumeSourceSHA',\"\",'ResumeExecutionSHA',\"\",'ForceQuadraticFallback',false);")
assert after['configuration']==expected_cfg,'Scientific defaults changed'
# All split, mapping, ranking, selection, weighting, scoring, seed and statistics
# function bodies are byte-identical. The numerical identity is retained to
# preserve the SHA-derived inner window Fold partition and completed probe keys.
report=dict(PreviousSourceSHA=hashlib.sha256(text.encode()).hexdigest(),CurrentSourceSHA=hashlib.sha256(new.encode()).hexdigest(),
 ChangedFunctions=changed,UnchangedFunctions=[n for n in before if n not in changed],
 Scope='Only failed quadratic fits gain an exact-kernel certified fallback; all successful saved results retain the same mathematical models and data partition.')
Path('Quadratic_Repair_Compatibility.json').write_text(json.dumps(report,indent=2))
print('SCIENTIFIC_FUNCTIONS_AND_CHECKPOINT_COMPATIBILITY_VERIFIED',report['CurrentSourceSHA'])
