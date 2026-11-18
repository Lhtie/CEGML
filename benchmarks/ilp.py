"""Finite, positive Horn ILP for HRI-inspired scaleup tasks.

The LLM supplies function-free rules, never executable Python. A synchronous,
Boolean forward-chaining interpreter shares the benchmark oracle contract.
"""
from functools import lru_cache
from itertools import product, count
import json
from pathlib import Path
import re

DATA = Path(__file__).resolve().parents[1]/'datasets/ilp'
PROMPT = '''Induce a positive function-free Horn program defining the target predicate from background knowledge and labeled ground atoms. The target's intended meaning is hidden; infer it from the examples.
{background}
Return clauses using lowercase predicate names and uppercase variables, e.g. head(X,Y) :- body1(X,Z), body2(Z,Y).
Each clause has one head and one or two body atoms, at most three distinct variables, and ends with a period. The target arity is declared with the background below; invented predicates may be unary or binary. Recursive and mutually recursive definitions are allowed. Background predicates are immutable. true and false are nullary Boolean atoms. Use the given relational predicates declared in the background. Do not use Python, arithmetic, numeric constants in rules, function terms, negation, disjunction inside clauses, cuts or external calls. Multiple clauses express alternative derivations. Practical parser budgets are 32 clauses and 12 invented predicates; these are our resource limits, not the HRI hierarchy constraint.
Inference is synchronous Boolean forward chaining from empty invented/target relations until the finite least fixed point. There is no fixed six/eight-round cutoff in this scaleup setting. Each IO input specifies N and a ground target atom. Output only the Horn program inside the requested ans block, with a concise reasoning summary.
{feedback}
Labeled ground atoms (JSON input => Boolean target truth):
{examples}
'''


def strip_fences(source):
    source = (source or '').strip()
    blocks = re.findall(r'```(?:prolog|datalog|horn|text)?\s*\n(.*?)```', source, re.S)
    return blocks[-1].strip() if blocks else source


def extract_candidate(response, default_extractor, multiple=False):
    if multiple:
        return [strip_fences(s) for s in default_extractor(response)]
    result = default_extractor(response)
    return strip_fences(result) if result is not None else None


def _atom(text):
    text = text.strip()
    if text in ('true', 'false'):
        return text, ()
    match = re.fullmatch(r'([a-z][a-z0-9_]*)\(\s*([A-Z][A-Za-z0-9_]*(?:\s*,\s*[A-Z][A-Za-z0-9_]*)?)\s*\)', text)
    if not match:
        raise ValueError('Expected a unary/binary atom with variables; arithmetic and numeric rule constants are not allowed')
    return match[1], tuple(v.strip() for v in match[2].split(','))


@lru_cache(maxsize=128)
def parse_program(source, schema=(), target_arity=1):
    source = strip_fences(source)
    source = re.sub(r'%[^\n]*|#[^\n]*', '', source).strip()
    if not source.endswith('.'):
        raise ValueError('Every Horn clause must end with a period')
    clauses = source.split('.')[:-1]
    if not 1 <= len(clauses) <= 32:
        raise ValueError('Expected 1..32 Horn clauses')
    arities = dict(schema) if schema else {'zero':1, 'succ':2}
    background = set(arities) | {'true','false'}
    arities.update({'true':0, 'false':0, 'target':target_arity})
    rules = []
    for clause in clauses:
        if clause.count(':-') != 1:
            raise ValueError('Expected head :- body. Ground facts and empty bodies are not used in this rule language')
        head, body = clause.split(':-')
        head = _atom(head)
        if head[0] in background:
            raise ValueError('Cannot redefine a background predicate')
        # Split at commas outside argument lists, rejecting leftovers explicitly.
        atoms = re.split(r',\s*(?![^()]*\))', body.strip())
        body = tuple(_atom(atom) for atom in atoms)
        if not 1 <= len(body) <= 2:
            raise ValueError('Expected one or two body atoms')
        variables = {v for _,args in (head,)+body for v in args}
        if len(variables) > 3:
            raise ValueError('A clause may use at most three variables')
        for pred,args in (head,)+body:
            if pred in arities and arities[pred] != len(args):
                raise ValueError(f'Inconsistent arity for {pred}')
            arities[pred] = len(args)
        rules.append((head, body))
    invented = set(arities)-background-{'target'}
    if len(invented) > 12:
        raise ValueError('At most 12 invented predicates')
    if not any(head[0] == 'target' for head,_ in rules):
        raise ValueError('Define the target predicate (use a false body for an empty relation)')
    return tuple(rules)


def _join(bindings, atom, relations):
    pred, variables = atom
    results = []
    # Index the relation on already bound arguments: a successor join should
    # not scan the whole relation for every binding at every recursive round.
    rows = relations.get(pred, ())
    indexes = {}
    for binding in bindings:
        positions = tuple(i for i, v in enumerate(variables) if v in binding)
        if positions not in indexes:
            index = {}
            for row in sorted(rows):
                index.setdefault(tuple(row[i] for i in positions), []).append(row)
            indexes[positions] = index
        key = tuple(binding[variables[i]] for i in positions)
        for row in indexes[positions].get(key, ()):
            updated = dict(binding)
            for variable,value in zip(variables,row):
                if variable in updated and updated[variable] != value:
                    break
                updated[variable] = value
            else:
                results.append(updated)
    return results


@lru_cache(maxsize=128)
def infer(source, num_constants, steps=None, background=None, target_arity=1):
    if not 1 <= num_constants <= 512 or (steps is not None and not 1 <= steps <= 1024):
        raise ValueError('Unsupported finite ILP domain/inference budget')
    if background is None:
        background = (('zero', ((0,),)), ('succ', tuple((i,i+1) for i in range(num_constants-1))))
    schema = tuple((pred, len(rows[0])) for pred,rows in background)
    rules = parse_program(source, schema, target_arity)
    domain = range(num_constants)
    relations = {pred:set(rows) for pred,rows in background}
    relations.update({'true':{()}, 'false':set()})
    for _ in (count() if steps is None else range(steps)):
        additions = {}
        for (pred,variables),body in rules:
            bindings = [{}]
            for atom in body:
                bindings = _join(bindings, atom, relations)
            head_rows = additions.setdefault(pred,set())
            for binding in bindings:
                free = sorted(set(variables)-binding.keys())
                for values in product(domain,repeat=len(free)):
                    grounded = {**binding, **dict(zip(free,values))}
                    head_rows.add(tuple(grounded[v] for v in variables))
        changed = False
        for pred,rows in additions.items():
            previous = relations.setdefault(pred,set())
            if not rows <= previous:
                previous.update(rows); changed = True
        if not changed:
            break
    return frozenset(row[0] if target_arity==1 else row for row in relations.get('target',set()))


class ILPProblem:
    def __init__(self, ident):
        from benchmarks.ilp_settings import SPECS
        if ident not in SPECS:
            raise ValueError('Unknown ILP setting: '+ident)
        self.ident = ident
        self.arity = SPECS[ident][1]
        self.data = json.loads((DATA/f'{ident}.json').read_text())
        runtime = self.data['scaleup']
        self.contexts = {runtime[k]['num_constants']:runtime[k] for k in ('train','eval')}
        self.backgrounds = {n:tuple((pred,tuple(tuple(row) for row in rows))
                                    for pred,rows in sorted(ctx['background'].items()))
                            for n,ctx in self.contexts.items()}
        self.train = [(row['input'],row['output']) for row in runtime['train']['examples']]
        self.eval = [(row['input'],row['output']) for row in runtime['eval']['examples']]
        self.labels = {json.dumps(x,sort_keys=True):y for x,y in self.train+self.eval}
        train_n, eval_n = (runtime[k]['num_constants'] for k in ('train','eval'))
        if ident in {'even_succ','predecessor','less_than','fizz','buzz'}:
            background = ('For each input N, constants are 0..N-1. zero(0) holds; '
                          'succ(i,j) holds exactly when j=i+1 and both are in the domain.')
            if ident == 'buzz':
                background += ' pred1(i,j) holds exactly when j=i+2; pred3(i,j) holds exactly when j=i+3, within the domain.'
        else:
            background = 'Closed-world background facts for each input num_constants value: '+json.dumps(
                {n:ctx['background'] for n,ctx in self.contexts.items()},separators=(',',':'))
        background += f' Target arity: target/{self.arity}. Training N={train_n}; evaluation/CE N={eval_n}. Background facts supply no target labels.'
        self.prompt = PROMPT.replace('{background}',background)
        self.icl_instruction = 'Infer Boolean truth of target ground atoms from the labeled examples.\n'+background
        self.settings = {k:self.data[k] for k in ('paper_url','author_repository','author_revision','author_task','method','paper_max_depth','paper_num_runs')}
        self.settings.update(setting=ident+'_scaleup',train_num_constants=train_n,eval_num_constants=eval_n,
                             train_examples=len(self.train),eval_examples=len(self.eval),target_arity=self.arity,
                             train_inference_steps=None,eval_inference_steps=None,
                             sample_schedule='shared_regex_geometric_scaling',default_start_size=3,default_ce_start_size=3,
                             clean_labels=True,hypothesis='positive_function_free_unary_binary_Horn',
                             data_sampling=self.data.get('data_sampling','deterministic_author_numeric_extension'),
                             hierarchy_bias='HRI_R_star_layer_parameterization_not_enforced',paper_optimizer_reproduced=False,
                             inference='synchronous_boolean_least_fixed_point',
                             parser_resource_limits={'max_clauses':32,'max_invented_predicates':12})

    def _query(self, x):
        if not isinstance(x,dict) or set(x) != {'num_constants','atom'}:
            raise ValueError('Expected {num_constants: N, atom: "target(...)"}')
        n = x['num_constants']
        if type(n) is not int or n not in self.contexts:
            raise ValueError('Unknown finite ILP background context')
        match = re.fullmatch(r'target\((\d+(?:,\d+)*)\)',x['atom']) if isinstance(x['atom'],str) else None
        args = tuple(map(int,match[1].split(','))) if match else ()
        if len(args)!=self.arity or any(i>=n for i in args):
            raise ValueError('Invalid ground target atom or arity')
        return n,args[0] if self.arity==1 else args

    def predict(self, source, x):
        n,atom = self._query(x)
        return atom in infer(source,n,None,self.backgrounds[n],self.arity)

    def reference_output(self, x):
        self._query(x)
        return self.labels[json.dumps(x,sort_keys=True)]

    def counterexamples(self, source, limit):
        pairs = [(x,y) for x,y in self.eval if self.predict(source,x) != y]
        eval_n = self.eval[0][0]['num_constants']
        details = {'backend':'finite_Horn_forward_chaining','policy':'ilp_scaleup_fixpoint_v2',
                   'guarantee':'finite_domain_least_fixed_point_equivalence',
                   'num_constants':eval_n,'inference_steps':None,'examples_checked':len(self.eval),
                   'disagreements':len(pairs),'domain':'all_target_ground_atoms_on_scaleup_evaluation_context',
                   'unbounded_rule_equivalence_proved':False}
        return pairs[:limit],not pairs,details
