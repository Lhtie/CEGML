"""Finite, positive Horn ILP for HRI-inspired scaleup tasks.

The LLM supplies function-free rules, never executable Python. A synchronous,
Boolean forward-chaining interpreter shares the benchmark oracle contract.
"""
from functools import lru_cache
from itertools import product, count
import json
from pathlib import Path
import re
from benchmarks.ilp_limits import Budget

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


def _join(bindings, atom, relations, budget=None):
    pred, variables = atom
    results = []
    # Index the relation on already bound arguments: a successor join should
    # not scan the whole relation for every binding at every recursive round.
    rows = relations.get(pred, ())
    indexes = {}
    for binding in bindings:
        positions = tuple(i for i, v in enumerate(variables) if v in binding)
        if len(positions)==len(variables):
            row=tuple(binding[v] for v in variables)
            if budget is not None: budget.step()
            if row in rows: results.append(dict(binding))
            continue
        if positions not in indexes:
            index = {}
            for row in sorted(rows):
                if budget is not None: budget.step()
                index.setdefault(tuple(row[i] for i in positions), []).append(row)
            indexes[positions] = index
        key = tuple(binding[variables[i]] for i in positions)
        for row in indexes[positions].get(key, ()):
            if budget is not None: budget.step()
            updated = dict(binding)
            for variable,value in zip(variables,row):
                if variable in updated and updated[variable] != value:
                    break
                updated[variable] = value
            else:
                results.append(updated)
                if budget is not None and len(results)>budget.max_bindings:
                    from benchmarks.ilp_limits import ResourceLimit
                    raise ResourceLimit("ILP intermediate binding budget exhausted")
    return results


def background_schema(background, declared=()):
    # Schema must survive an empty relation. Known task predicates have fixed
    # signatures; arbitrary callers can supply an explicit declaration.
    known = {'zero':1,'succ':2,'pred1':2,'pred3':2,'edge':2,'color':2,
             'father':2,'mother':2,'son':2,'daughter':2,'const':2,'red':1,'neq':2,'parent':2,'value':2,'brother':2,'sister':2}
    known.update(dict(declared))
    schema = []
    for pred,rows in background:
        arity = known.get(pred, len(rows[0]) if rows else None)
        if arity is None:
            raise ValueError(f'Empty background {pred} needs an explicit arity')
        if any(len(row)!=arity for row in rows):
            raise ValueError(f'Invalid background arity for {pred}')
        schema.append((pred,arity))
    return tuple(schema)


def _acyclic_rules(rules,edb):
    # Explicit dependency graph; no rule evaluation occurs here.
    graph={p:set() for (p,_),_ in rules}
    for (p,_),body in rules:graph[p].update(q for q,_ in body if q not in edb)
    visiting=set();done=set()
    def visit(p):
        if p in visiting:return False
        if p in done:return True
        visiting.add(p)
        if not all(visit(q) for q in graph.get(p,())):return False
        visiting.remove(p);done.add(p);return True
    return all(visit(p) for p in list(graph))


def _infer_acyclic(rules,n,background,arity,budget):
    # Independent backward interpreter for nonrecursive rules. In particular,
    # existential neq joins need not materialize every auxiliary pair.
    from functools import lru_cache
    edb={p:set(rows) for p,rows in background};edb.update(true={()},false=set())
    by_head={p:[] for (p,_),_ in rules}
    for head,body in rules:by_head[head[0]].append((head,body))
    @lru_cache(maxsize=100000)
    def holds(pred,args):
        budget.step()
        if pred in edb:return args in edb[pred]
        for (_,head),body in by_head.get(pred,()):
            initial={}
            for var,value in zip(head,args):
                if var in initial and initial[var]!=value:break
                initial[var]=value
            else:
                bindings=[initial]
                for atom in body:
                    if atom[0] in edb:bindings=_join(bindings,atom,edb,budget)
                variables={v for _,av in body for v in av}
                for binding in bindings:
                    free=sorted(variables-binding.keys())
                    for values in product(range(n),repeat=len(free)):
                        budget.step();env={**binding,**dict(zip(free,values))}
                        if all(holds(p,tuple(env[v] for v in av)) for p,av in body):return True
        return False
    return frozenset(args[0] if arity==1 else args for args in product(range(n),repeat=arity) if holds('target',args))


@lru_cache(maxsize=128)
def infer(source, num_constants, steps=None, background=None, target_arity=1, schema=()):
    if not 1 <= num_constants <= 512 or (steps is not None and not 1 <= steps <= 1024):
        raise ValueError('Unsupported finite ILP domain/inference budget')
    if background is None:
        background = (('zero', ((0,),)), ('succ', tuple((i,i+1) for i in range(num_constants-1))))
    schema = background_schema(background, schema)
    budget = Budget()
    rules = parse_program(source, schema, target_arity)
    if steps is None and _acyclic_rules(rules,{p for p,_ in background}|{'true','false'}):
        return _infer_acyclic(rules,num_constants,background,target_arity,budget)
    domain = range(num_constants)
    relations = {pred:set(rows) for pred,rows in background}
    relations.update({'true':{()}, 'false':set()})
    for _ in (count() if steps is None else range(steps)):
        budget.check()
        additions = {}
        for (pred,variables),body in rules:
            bindings = [{}]
            for atom in body:
                bindings = _join(bindings, atom, relations, budget)
            head_rows = additions.setdefault(pred,set())
            for binding in bindings:
                free = sorted(set(variables)-binding.keys())
                for values in product(domain,repeat=len(free)):
                    budget.step()
                    grounded = {**binding, **dict(zip(free,values))}
                    head_rows.add(tuple(grounded[v] for v in variables))
        changed = False
        for pred,rows in additions.items():
            previous = relations.setdefault(pred,set())
            if not rows <= previous:
                previous.update(rows); changed = True
        budget.facts(sum(map(len,relations.values())))
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
        if ident in {'even_succ','even_odd','predecessor','less_than','fizz','buzz'}:
            background = ('For each input N, constants are 0..N-1. zero(0) holds; '
                          'succ(i,j) holds exactly when j=i+1 and both are in the domain.')
            if ident == 'buzz':
                background += ' pred1(i,j) holds exactly when j=i+2; pred3(i,j) holds exactly when j=i+3, within the domain.'
        else:
            background = 'Closed-world background facts for each input num_constants value: '+json.dumps(
                {n:{pred:rows for pred,rows in ctx['background'].items() if pred!='neq'} for n,ctx in self.contexts.items()},separators=(',',':'))
            if ident=='two_children':
                background += ' For each domain, neq(i,j) holds exactly when i and j are distinct constants.'
        background += f' Target arity: target/{self.arity}. Training N={train_n}; evaluation/CE N={eval_n}. Background facts supply no target labels.'
        self.prompt = PROMPT.replace('{background}',background)
        self.icl_instruction = 'Infer Boolean truth of target ground atoms from the labeled examples.\n'+background
        self.settings = {k:self.data[k] for k in ('paper_url','author_repository','author_revision','author_task','method','paper_max_depth','paper_num_runs')}
        self.settings.update(oracle_policy='finite_Horn_ranked_smt_v1', oracle_backend='cvc5_python_api',
                             oracle_contexts=['train','eval'],oracle_semantics='finite_least_fixed_point',
                             oracle_timeout_ms_per_check=10000,oracle_wall_seconds=30,
                             setting=ident+'_scaleup',train_num_constants=train_n,eval_num_constants=eval_n,
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
        from benchmarks.ilp_smt import search
        return search(source, self, limit=limit)
