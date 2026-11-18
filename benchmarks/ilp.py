"""Finite, positive Horn ILP with the author's EvenSucc facts and labels.

The LLM supplies function-free rules, never executable Python. A synchronous,
Boolean forward-chaining interpreter shares the benchmark oracle contract.
"""
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import re

DATA = Path(__file__).resolve().parents[1]/'datasets/ilp'
PROMPT = '''Induce a positive function-free Horn program defining the unary predicate target/1 from background knowledge and labeled ground atoms. The target's intended meaning is hidden; infer it from the examples.
{background}
Return clauses using lowercase predicate names and uppercase variables, e.g. head(X,Y) :- body1(X,Z), body2(Z,Y).
Each clause has one head and one or two body atoms, at most three distinct variables, and ends with a period. target is unary; invented predicates may be unary or binary. Recursive and mutually recursive definitions are allowed. Background predicates are immutable. true and false are nullary Boolean atoms. The only given relational predicates are zero/1 and succ/2. Do not use Python, arithmetic, numeric constants in rules, function terms, negation, disjunction inside clauses, cuts or external calls. Multiple clauses express alternative derivations. Practical parser budgets are 32 clauses and 12 invented predicates; these are our resource limits, not the HRI hierarchy constraint.
Inference is synchronous Boolean forward chaining from empty invented/target relations, with six rounds for N=11 and eight rounds for N=15. Each IO input specifies N and a ground target atom. Output only the Horn program inside the requested ans block, with a concise reasoning summary.
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
def parse_program(source):
    source = strip_fences(source)
    source = re.sub(r'%[^\n]*|#[^\n]*', '', source).strip()
    if not source.endswith('.'):
        raise ValueError('Every Horn clause must end with a period')
    clauses = source.split('.')[:-1]
    if not 1 <= len(clauses) <= 32:
        raise ValueError('Expected 1..32 Horn clauses')
    arities = {'zero':1, 'succ':2, 'true':0, 'false':0, 'target':1}
    background = {'zero','succ','true','false'}
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
        raise ValueError('Define target/1 (use target(X) :- false. for an empty target relation)')
    return tuple(rules)


def _join(bindings, atom, relations):
    pred, variables = atom
    results = []
    for binding in bindings:
        for row in sorted(relations.get(pred, ())):
            updated = dict(binding)
            for variable,value in zip(variables,row):
                if variable in updated and updated[variable] != value:
                    break
                updated[variable] = value
            else:
                results.append(updated)
    return results


@lru_cache(maxsize=128)
def infer(source, num_constants, steps):
    if not 1 <= num_constants <= 15 or not 1 <= steps <= 8:
        raise ValueError('Unsupported finite ILP domain/inference budget')
    rules = parse_program(source)
    domain = range(num_constants)
    relations = {'zero':{(0,)}, 'succ':{(i,i+1) for i in range(num_constants-1)},
                 'true':{()}, 'false':set()}
    for _ in range(steps):
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
    return frozenset(row[0] for row in relations.get('target',set()))


class ILPProblem:
    def __init__(self, ident):
        if ident != 'even_succ':
            raise ValueError('Unknown ILP setting; available: even_succ')
        self.data = json.loads((DATA/f'{ident}.json').read_text())
        self.contexts = {self.data[k]['num_constants']:self.data[k] for k in ('train','eval')}
        self.train = [(row['input'],row['output']) for row in self.data['train']['examples']]
        self.eval = [(row['input'],row['output']) for row in self.data['eval']['examples']]
        self.labels = {json.dumps(x,sort_keys=True):y for x,y in self.train+self.eval}
        initial = self.data['train']
        background = (f"Training constants: 0..{initial['num_constants']-1}. "
                      f"Background facts: {json.dumps(initial['background'])}.\n"
                      "For each input's num_constants=N, constants are 0..N-1; zero holds only at 0, "
                      "succ(i,j) holds exactly when j=i+1 and both are in the domain. "
                      "The same background declaration extends to N=15 during evaluation or CE feedback. "
                      "Only background facts, never target labels, are supplied by this declaration.")
        self.prompt = PROMPT.replace('{background}',background)
        self.icl_instruction = ('Infer Boolean truth of target ground atoms from the labeled examples and this relational background.\n'+background)
        self.settings = {k:self.data[k] for k in ('paper_url','author_repository','author_revision','author_task','method','paper_max_depth','paper_num_runs')}
        self.settings.update(train_num_constants=11, eval_num_constants=15,
                             train_inference_steps=6, eval_inference_steps=8,
                             clean_labels=True, hypothesis='positive_function_free_unary_binary_Horn',
                             hierarchy_bias='HRI_R_star_layer_parameterization_not_enforced',
                             paper_optimizer_reproduced=False, inference='synchronous_boolean_forward_chaining',
                             parser_resource_limits={'max_clauses':32,'max_invented_predicates':12})

    def _query(self, x):
        if not isinstance(x,dict) or set(x) != {'num_constants','atom'}:
            raise ValueError('Expected {num_constants: N, atom: "target(i)"}')
        n = x['num_constants']
        if type(n) is not int or n not in self.contexts:
            raise ValueError('Expected paper context N=11 or N=15')
        match = re.fullmatch(r'target\((\d+)\)', x['atom']) if isinstance(x['atom'],str) else None
        if not match or not 0 <= int(match[1]) < n:
            raise ValueError('Ground atom outside the declared domain')
        return n,int(match[1])

    def predict(self, source, x):
        n,i = self._query(x)
        return i in infer(source,n,self.contexts[n]['inference_steps'])

    def reference_output(self, x):
        self._query(x)
        return self.labels[json.dumps(x,sort_keys=True)]

    def counterexamples(self, source, limit):
        parse_program(source)
        pairs = [(x,y) for x,y in self.eval if self.predict(source,x) != y]
        details = {'backend':'finite_Horn_forward_chaining', 'policy':'hri_even_succ_v1',
                   'guarantee':'finite_domain_bounded_inference_equivalence',
                   'num_constants':15, 'inference_steps':8, 'examples_checked':len(self.eval),
                   'disagreements':len(pairs), 'domain':'all_target_ground_atoms_on_author_evaluation_context',
                   'unbounded_rule_equivalence_proved':False}
        return pairs[:limit],not pairs,details
