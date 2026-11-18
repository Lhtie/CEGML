"""Reconstruct the deterministic HRI EvenSucc data without a torch dependency."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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
    (ROOT/'even_succ.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print('Prepared EvenSucc: train 11 constants/6 steps; eval 15 constants/8 steps')


if __name__ == '__main__': main()
