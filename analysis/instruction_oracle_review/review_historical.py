import json,collections
from pathlib import Path
from benchmarks.instruction_oracle import reference,parse_verdict,equivalent_answer,normalize,input_key
root=Path('logs/benchmarks/icl_gen_instruction_induction/model=gpt-oss')
out=Path('analysis/instruction_oracle_review');records=[];counts=collections.Counter();task_summaries=[]
for mode in ('std','ce'):
 for p in sorted((root/mode/'reg').glob('*.json')):
  data=json.loads(p.read_text());name=p.stem;ref=reference(name)
  for runid,rows in enumerate(data.get('runs',[])):
   for row in rows:
    record={'task':name,'method':mode,'epoch':row.get('epoch'),'source_log':str(p),'hypothesis':row.get('Prediction'),'old_equivalent':row.get('Equivalent'),'human_annotations':ref['annotations'],'issues':[]}
    counts['epoch_records']+=1
    try: verdict=parse_verdict(row.get('OracleResponse'))
    except ValueError:
     record['issues'].append('missing_or_invalid_oracle_response');counts['missing_or_invalid_response']+=1;records.append(record);continue
    proposals=verdict['counterexamples'];record['raw_proposals']=proposals;record['oracle_reason']=verdict.get('reason')
    if name=='cause_and_effect' and 'Sentence 1:' not in row.get('Prompt',''):
     record['issues'].append('historical_task_adapter_changed_problem');counts['historical_task_adapter_changed_problem']+=1
    for i,ce in enumerate(proposals):
     counts['proposed_counterexamples']+=1;issues=[];x=ce.get('input');y=ce.get('expected_output');guess=ce.get('candidate_output')
     if name!='common_concept' and not isinstance(x,str):issues.append('wrong_input_schema')
     if guess is not None and equivalent_answer(name,y,guess):issues.append('no_output_difference_under_task_metric')
     accepted=ref['known'].get(input_key(x),[])
     if guess is not None and any(equivalent_answer(name,guess,v) for v in accepted):issues.append('candidate_is_known_accepted_alternative')
     if accepted and not any(equivalent_answer(name,y,v) for v in accepted):issues.append('expected_label_conflicts_with_induction_reference')
     if issues:record['issues'].append({'proposal_index':i,'flags':issues,'proposal':ce})
     for issue in issues:counts[issue]+=1
    records.append(record)
for p in sorted(Path('datasets/instruction_induction/annotations').glob('*.json')):
 task_summaries.append({'task':p.stem,'official_annotations':reference(p.stem)['annotations'],'hypotheses':[r for r in records if r['task']==p.stem]})
(out/'historical_audit.json').write_text(json.dumps({'counts':dict(counts),'tasks':task_summaries},ensure_ascii=False,indent=2))
print(dict(counts))
