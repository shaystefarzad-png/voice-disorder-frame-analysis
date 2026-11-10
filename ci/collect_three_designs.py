"""Collect existing completed pilot reports; do not train or select new models."""
import json, math
from pathlib import Path
from openpyxl import load_workbook, Workbook

root = Path('downloaded-three-designs')
out = Path('three-designs-pilot-summary')
out.mkdir(exist_ok=True)
rows = []
statistics = []
timing = []
for disease in range(1, 7):
    directory = root / f'three-designs-pilot-d{disease}'
    files = list(directory.rglob('All_Diseases_Summary.xlsx'))
    assert len(files) == 1, (disease, files)
    workbook = load_workbook(files[0], read_only=True, data_only=True)
    data = list(workbook['Summary'].values)
    keys = data[0]
    group = [dict(zip(keys, row)) for row in data[1:] if any(v is not None for v in row)]
    assert len(group) == 12 and all(row['Repeat'] == 1 for row in group)
    rows += group
    statfiles = list(directory.rglob('Statistical_Hypothesis_Validation.xlsx'))
    assert len(statfiles) == 1
    data = list(load_workbook(statfiles[0], read_only=True, data_only=True)['PerDisease'].values)
    statistics += [dict(zip(data[0], row)) for row in data[1:] if any(v is not None for v in row)]
    timefiles = list(directory.rglob('PilotTiming.json'))
    assert len(timefiles) == 1
    timing.append(json.loads(timefiles[0].read_text()))
assert len(rows) == 72 and len(set(r['Disease'] for r in rows)) == 6
assert all(0 <= r['TestAccuracy'] <= 1 and 0 <= r['TestSensitivity'] <= 1 for r in rows)
comparisons = []
for row in rows:
    if row['Scenario'] not in ['S3A_Hybrid', 'S3B_Voting']:
        continue
    for single in ['S1_25ms', 'S2_75ms']:
        base = next(r for r in rows if r['Disease'] == row['Disease'] and r['Design'] == row['Design'] and r['Scenario'] == single)
        comparisons.append(dict(Disease=row['Disease'], Design=row['Design'], Hybrid=row['Scenario'], SingleWindow=single,
                                DeltaAccuracy=row['TestAccuracy'] - base['TestAccuracy'],
                                DeltaSensitivity=row['TestSensitivity'] - base['TestSensitivity'],
                                DeltaBalancedAccuracy=row['TestBalancedAccuracy'] - base['TestBalancedAccuracy']))
report = dict(SourceRun=42849039670, Repeats=1, IndependentHypothesisConfirmation=False,
              Interpretation='Exploratory internal validation after prior Test inspection; one repeat does not establish statistical significance.',
              Summary=rows, Comparisons=comparisons, Statistics=statistics, Timing=timing)
(out / 'Pilot_Report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
workbook = Workbook()
workbook.remove(workbook.active)
for name, records in [('Summary', rows), ('PairedComparisons', comparisons), ('Statistics', statistics), ('Timing', timing)]:
    ws = workbook.create_sheet(name)
    if records:
        headers = list(records[0]);ws.append(headers)
        for record in records:ws.append([record.get(k) for k in headers])
        ws.freeze_panes = 'A2';ws.auto_filter.ref = ws.dimensions
workbook.save(out / 'All_Six_Diseases_Pilot_Summary.xlsx')
print('Collected all 72 subject-level scenario reports from the completed pilot, no retraining.')
