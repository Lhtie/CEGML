"""Reproducible, setting-level difficulty proxies; no model dependencies.

All local IDs are annotated. Equal rank bands are sampling strata, not proofs
of semantic minimality or empirically calibrated LLM difficulty.
"""
from functools import lru_cache
from pathlib import Path
from collections import Counter
import argparse
import ast
import hashlib
import json
import string

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'benchmark_difficulty_v2'
CATALOG = ROOT/'datasets/benchmarks/difficulty_catalog.json'

INSTRUCTION_TIERS = (
    ('diff','sum','first_word_letter','second_word_letter','letters_list','orthography_starts_with','num_to_verbal','taxonomy_animal'),
    ('singular_to_plural','antonyms','synonyms','rhymes','negation','larger_animal','sentiment','word_in_context'),
    ('active_to_passive','cause_and_effect','common_concept','informal_to_formal','sentence_similarity','translation_en-de','translation_en-es','translation_en-fr'),
)
SCAN_TIERS = (
    ('simple','addprim_turn_left'),
    ('addprim_jump','template_right'),
    ('template_opposite_right','template_around_right'),
    ('length','template_jump_around_right'),
)
ILP_TIERS = (
    ('has_father','predecessor','undirected_edge','less_than'),
    ('grandparent','graph_coloring','connectedness','cyclic','adj_to_red','two_children','son','grandparent_nlm'),
    ('even_succ','buzz','fizz','length','member','relatedness','even_odd'),
)
SCHEMES = {
 'cryptography': {'levels':3,'axis':'training_information_coverage',
   'basis':'Within each cipher family rank clean ten-example training sets by missing lowercase-letter coverage, then missing case-sensitive coverage; higher is harder. Rotate remainder bands across families to make 100 IDs per level overall.',
   'limitations':'Information/identifiability proxy, not cipher computational complexity. Same substitution structure within a family; rank ties may cross levels. Applies to default clean paper setting with zero OOD ratio.'},
 'list_functions': {'levels':5,'axis':'reference_syntax_complexity',
   'basis':'Weighted AST score: nodes +8*loops +4*conditions +3*calls +2*subscripts +5*comprehensions; sort by (score,ID), five bands of 50.',
   'limitations':'Representation-dependent rank proxy, not normalized DSL composition depth. Equal scores may cross level boundaries.'},
 'instruction_induction': {'levels':3,'axis':'task_contract_and_semantic_requirements',
   'basis':'L1 explicit extraction/arithmetic/category rules; L2 lexical exceptions/contextual decisions; L3 sentence relations/rewrites, semantic ratings and translation. Eight fixed task IDs per level.',
   'limitations':'Manual sampling proxy, not measured model difficulty. Different task families have incomparable priors and language expertise.'},
 'scan': {'levels':4,'axis':'generalization_protocol',
   'basis':'All official split IDs, ranked by output-length extrapolation gap, unseen compositional jump coverage, missing unique training commands, then ID; four rank bands.',
   'limitations':'Heuristic generalization-gap strata, not a single command-depth ordering. Every split still targets the full SCAN interpreter/oracle.'},
 'ilp': {'levels':3,'axis':'relational_structure_and_recursive_construction',
   'basis':'L1 direct relations/ordered-chain closure; L2 relational joins/general-graph paths/cycles; L3 recursive numerical/list construction. Retain all 19 native configurations: 4/8/7, without dropping or duplicating IDs.',
   'limitations':'Manual mechanism proxy, not a recursion-free/recursive dichotomy or proven minimum rule depth. Includes current author-code unary GraphColoring and scaleup samplers.'},
}


def ast_features(source):
    nodes=list(ast.walk(ast.parse(source)))
    counts={'nodes':len(nodes),'loops':sum(isinstance(n,(ast.For,ast.While)) for n in nodes),
            'conditions':sum(isinstance(n,(ast.If,ast.IfExp)) for n in nodes),
            'calls':sum(isinstance(n,ast.Call) for n in nodes),
            'subscripts':sum(isinstance(n,ast.Subscript) for n in nodes),
            'comprehensions':sum(isinstance(n,ast.comprehension) for n in nodes)}
    counts['score']=counts['nodes']+8*counts['loops']+4*counts['conditions']+3*counts['calls']+2*counts['subscripts']+5*counts['comprehensions']
    return counts


def build_catalog():
    entries=[]; fingerprints={}
    for definition in (Path(__file__),ROOT/'benchmarks/ilp_settings.py',ROOT/'benchmarks/ilp_author_data.py'):
        fingerprints[str(definition.relative_to(ROOT))]=hashlib.sha256(definition.read_bytes()).hexdigest()
    def read(path):
        raw=path.read_bytes();fingerprints[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
        return raw.decode()
    def add(task,ident,level,features):
        entries.append({'task':task,'id':ident,'tier':f'L{level}','level':level,
                        'levels':SCHEMES[task]['levels'],'axis':SCHEMES[task]['axis'],
                        'proxy':True,'features':features})
    for family_index,family in enumerate(('caesar','atbash','keyboard')):
        rows=[json.loads(line) for line in read(ROOT/f'datasets/cryptography/raw/{family}.jsonl').splitlines() if line.strip()]
        ranked=[]
        for i,row in enumerate(rows):
            letters={c for pair in row['train']['normal'] for c in pair['input'] if c in string.ascii_letters}
            features={'missing_lowercase_letters':26-len({c.lower() for c in letters}),
                      'missing_case_sensitive_letters':52-len(letters),'family':family,
                      'training_entries':len(row['train']['normal'])}
            ranked.append((features['missing_lowercase_letters'],features['missing_case_sensitive_letters'],i,features))
        ranked.sort(key=lambda r:r[:3])
        q,remainder=divmod(len(ranked),3)
        sizes=[q+(int((level-family_index)%3<remainder)) for level in range(3)]
        cursor=0
        for level,size in enumerate(sizes,1):
            for rank,(_,_,i,features) in enumerate(ranked[cursor:cursor+size],cursor+1):
                add('cryptography',f'{family}:{i}',level,{**features,'within_family_rank':rank})
            cursor+=size
    rows=[json.loads(line) for line in read(ROOT/'datasets/list_functions/raw/list_functions.jsonl').splitlines() if line.strip()]
    ranked=sorted((ast_features(row['function'])['score'],row['id'],ast_features(row['function'])) for row in rows)
    for i,(_,ident,features) in enumerate(ranked):
        add('list_functions',ident,min(5,i*5//len(ranked)+1),{**features,'population_rank':i+1})
    # Full official SCAN inventory: every train file must have its test pair.
    universe={line.split(' OUT: ')[0] for line in read(ROOT/'datasets/scan/raw/tasks.txt').splitlines()}
    scan_ranked=[]
    source_manifest=json.loads(read(ROOT/'datasets/scan/source_manifest.json'))
    native_paths={Path(r['local_file']).stem.removeprefix('tasks_train_'):r['source_path']
                  for r in source_manifest['files'] if r['local_file'].startswith('tasks_train_')}
    for path in sorted((ROOT/'datasets/scan/raw').glob('tasks_train_*.txt')):
        ident=path.stem.removeprefix('tasks_train_')
        test=path.with_name('tasks_test_'+ident+'.txt')
        if not test.is_file():raise ValueError('Missing SCAN test split: '+ident)
        train_rows=read(path).splitlines();test_rows=read(test).splitlines()
        train_commands={line.split(' OUT: ')[0] for line in train_rows}
        train_max=max(len(line.split(' OUT: ')[1].split()) for line in train_rows)
        test_max=max(len(line.split(' OUT: ')[1].split()) for line in test_rows)
        jump_count=sum('jump' in cmd.split() and cmd!='IN: jump' for cmd in train_commands)
        all_jump=sum('jump' in cmd.split() and cmd!='IN: jump' for cmd in universe)
        features={'output_length_gap':max(0,test_max-train_max),
                  'unseen_compositional_jump':all_jump-jump_count,
                  'missing_unique_training_commands':len(universe-train_commands),
                  'train_rows':len(train_rows),'unique_train_commands':len(train_commands),
                  'test_rows':len(test_rows),'source_path':native_paths[ident]}
        scan_ranked.append((features['output_length_gap'],features['unseen_compositional_jump'],
                            features['missing_unique_training_commands'],ident,features))
    scan_ranked.sort(key=lambda row:row[:4])
    for i,(*_,ident,features) in enumerate(scan_ranked):
        add('scan',ident,min(4,i*4//len(scan_ranked)+1),{**features,'population_rank':i+1})
    # Hand annotations must cover the actual local IDs exactly, not silently skip new ones.
    for task,tiers,paths in (
        ('instruction_induction',INSTRUCTION_TIERS,list((ROOT/'datasets/instruction_induction/raw/induce').glob('*.json'))),
        ('ilp',ILP_TIERS,list((ROOT/'datasets/ilp').glob('*.json')))):
        actual=set()
        for path in sorted(paths):
            if task=='instruction_induction':
                read(path);ident=path.stem
            else:
                fingerprints[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                ident=path.stem
            actual.add(ident)
        annotated=[ident for tier in tiers for ident in tier]
        if len(set(annotated))!=len(annotated) or set(annotated)!=actual:
            raise ValueError(f'{task}: difficulty annotations do not match local IDs: {actual^set(annotated)}')
        for level,tier in enumerate(tiers,1):
            for ident in tier:add(task,ident,level,{'annotation':'manual_a_priori'})
    entries.sort(key=lambda e:(e['task'],e['level'],e['id']))
    if len({(e['task'],e['id']) for e in entries})!=len(entries):raise ValueError('Duplicate setting IDs')
    return {'version':VERSION,'scope':'all_local_setting_IDs_not_IO_resampling','empirically_calibrated':False,
            'schemes':SCHEMES,'source_sha256':fingerprints,
            'counts':{task:dict(sorted(Counter(e['tier'] for e in entries if e['task']==task).items())) for task in SCHEMES},
            'settings':entries}


@lru_cache(maxsize=1)
def catalog():
    return json.loads(CATALOG.read_text())


@lru_cache(maxsize=1)
def _lookup():
    return {(e['task'],e['id']):e for e in catalog()['settings']}


def annotation(task,ident):
    if task=='cryptography' and ':' not in ident:ident+=':0'
    entry=_lookup().get((task,ident))
    if entry is None:raise ValueError(f'Missing difficulty metadata: {task}/{ident}; regenerate benchmarks.difficulty')
    return {'catalog_version':VERSION,**{k:v for k,v in entry.items() if k!='task'}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(CATALOG))
    args=parser.parse_args(argv)
    result=build_catalog();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'settings':len(result['settings']),'counts':result['counts'],'output':str(path)},ensure_ascii=False))


if __name__=='__main__':main()
