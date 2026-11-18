"""Reconstruct the deterministic HRI EvenSucc data without a torch dependency."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from benchmarks.ilp_settings import SPECS, REFERENCES, make_split
REVISION = '1a3a0465a5a5305ad8fec958901710dc6ead76e4'
REFERENCE_PROGRAM = '''target(X) :- zero(X).
aux(X,Y) :- succ(X,Z), succ(Z,Y).
target(X) :- target(Y), aux(Y,X).'''


def split(n, steps):
    return {'num_constants': n, 'inference_steps': steps,
            'background': {'zero': [[0]], 'succ': [[i, i+1] for i in range(n-1)]},
            'examples': [{'input': {'num_constants': n, 'atom': f'target({i})'},
                          'output': i % 2 == 0} for i in range(n)]}


def main():
    data = {'id': 'even_succ', 'paper': 'Glanois et al., ICML 2022',
            'paper_url': 'https://proceedings.mlr.press/v162/glanois22a.html',
            'author_repository': 'https://github.com/claireaoi/hierarchical-rule-induction',
            'author_revision': REVISION, 'author_task': 'EvenSucc',
            'train': split(11, 6), 'eval': split(15, 8),
            'reference_program': REFERENCE_PROGRAM,
            'paper_max_depth': 4, 'paper_num_runs': 10,
            'method': 'LLM_CE_adaptation_not_HRI_optimizer',
            'source_files': {str(p.relative_to(ROOT/'source')): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((ROOT/'source').rglob('*')) if p.is_file()}}
    data['scaleup'] = {'train': split(384, None), 'eval': split(512, None),
                       'inference': 'finite_least_fixed_point',
                       'paper_setting': False}
    (ROOT/'even_succ.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    for ident, (author_task, arity, meaning) in SPECS.items():
        if ident == 'even_succ':
            continue
        item = {k:v for k,v in data.items() if k not in {'train','eval','scaleup','id','reference_program','author_task'}}
        train_n, eval_n = (384,512) if arity==1 else (20,24)
        item.update(id=ident,author_task=author_task,target_meaning=meaning,
                    reference_program=REFERENCES[ident],
                    data_sampling='seeded_sparse_scaleup_adaptation_not_author_random_sampler',
                    scaleup={'train':make_split(ident,train_n,43,384),
                             'eval':make_split(ident,eval_n,1043),
                             'inference':'finite_least_fixed_point','paper_setting':False})
        # Root markers retain compatibility with suite data discovery.
        item['train'] = item['scaleup']['train']
        item['eval'] = item['scaleup']['eval']
        (ROOT/f'{ident}.json').write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n')
    print(f'Prepared {len(SPECS)} ILP scaleup tasks; balanced selects six; each has 384 training atoms.')


if __name__ == '__main__': main()
