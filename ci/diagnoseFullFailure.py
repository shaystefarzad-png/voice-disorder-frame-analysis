import json,pathlib,subprocess,re,traceback
root=pathlib.Path('recovery/full-run-diagnosis');root.mkdir(parents=True,exist_ok=True)
try:
    r=subprocess.run(['gh','api','--paginate','--slurp','repos/shaystefarzad-png/voice-disorder-frame-analysis/actions/runs/42255192906/jobs?per_page=100'],capture_output=True,text=True,timeout=150)
    if r.returncode:raise RuntimeError(r.stderr)
    jobs=[j for page in json.loads(r.stdout) for j in page['jobs'] if j['conclusion']=='failure']
    (root/'failed-jobs.jsonl').write_text('\n'.join(json.dumps({k:j[k] for k in ['id','name','conclusion','steps']}) for j in jobs))
    for job in jobs:
        r=subprocess.run(['gh','api',f'repos/shaystefarzad-png/voice-disorder-frame-analysis/actions/jobs/{job["id"]}/logs'],capture_output=True,text=True,timeout=120)
        lines=(r.stdout if r.returncode==0 else r.stderr).splitlines();keep=set()
        for i,line in enumerate(lines):
            if re.search(r'Error using|Error in |Required candidate|Certified SVM|##\[error\]|timed out|exceeded|assert|converge|Exception|terminated',line,re.I):keep.update(range(max(0,i-2),min(len(lines),i+12)))
        text='\n'.join(lines[i] for i in sorted(keep)) if keep else '\n'.join(lines[-70:])
        (root/f'job-{job["id"]}-error.txt').write_text(text)
except Exception:
    text=traceback.format_exc();(root/'diagnostic-error.txt').write_text(text)
    print('::error title=Exact diagnosis retrieval::'+text.replace('%','%25').replace('\n','%0A'))
