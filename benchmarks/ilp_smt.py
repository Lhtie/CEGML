"""cvc5 finite positive-Horn oracle with exact least-model rank constraints.

A relevance grounder constructs ground rule supports, not verdicts or labels.
SMT enforces rule closure AND well-founded support; Horn implications alone
would incorrectly admit self-supported facts. Acceptance is finite-context
only. Resource exhaustion remains unknown; SAT witnesses are replayed.
"""
from collections import defaultdict
from itertools import product
from time import monotonic
import cvc5
from cvc5 import Kind as K
from benchmarks.ilp_limits import Budget, ResourceLimit, execution_deadline

# Shared ownership, matching crypto_smt: terms must not outlive their manager.
_TERM_MANAGER = cvc5.TermManager()


def _acyclic(rules, edb):
    graph=defaultdict(set)
    for (head,_),body in rules:
        graph[head].update(p for p,_ in body if p not in edb)
    visiting=set();done=set()
    def visit(p):
        if p in visiting:return False
        if p in done:return True
        visiting.add(p)
        if not all(visit(q) for q in graph.get(p,())):return False
        visiting.remove(p);done.add(p);return True
    return all(visit(p) for p in list(graph))


def _query_ground(rules,n,background,roots,budget):
    # Nonrecursive rules can be grounded backward from target queries. This
    # avoids materializing every auxiliary pair in neq/color joins.
    relations={p:set(rows) for p,rows in background};relations.update(true={()},false=set())
    indexes={};by_head=defaultdict(list)
    for head,body in rules:by_head[head[0]].append((head,body))
    supports={};todo=list(roots)
    def join(bindings,atom):
        pred,variables=atom;out=[];rows=relations[pred]
        for binding in bindings:
            positions=tuple(i for i,v in enumerate(variables) if v in binding)
            key=tuple(binding[variables[i]] for i in positions)
            if len(positions)==len(variables): candidates=(key,) if key in rows else ()
            else:
                cache_key=(pred,positions)
                if cache_key not in indexes:
                    index=defaultdict(list)
                    for row in rows:
                        budget.step();index[tuple(row[i] for i in positions)].append(row)
                    indexes[cache_key]=index
                candidates=indexes[cache_key].get(key,())
            for row in candidates:
                budget.step();updated=dict(binding)
                for var,value in zip(variables,row):
                    if var in updated and updated[var]!=value:break
                    updated[var]=value
                else:out.append(updated)
                if len(out)>budget.max_bindings:raise ResourceLimit('ILP intermediate binding budget exhausted')
        return out
    edb=set(relations)
    while todo:
        budget.step();atom=todo.pop()
        if atom in supports:continue
        pred,args=atom;supports[atom]=set()
        for (_,head_args),body in by_head.get(pred,()):
            initial={}
            for var,value in zip(head_args,args):
                if var in initial and initial[var]!=value:break
                initial[var]=value
            else:
                bindings=[initial]
                for item in body:
                    if item[0] in edb:bindings=join(bindings,item)
                all_vars={v for _,av in body for v in av}
                for binding in bindings:
                    free=sorted(all_vars-binding.keys())
                    for values in product(range(n),repeat=len(free)):
                        budget.step();env={**binding,**dict(zip(free,values))}
                        deps=tuple(sorted(set((p,tuple(env[v] for v in av)) for p,av in body if p not in edb)))
                        supports[atom].add(deps)
                        todo.extend(d for d in deps if d not in supports)
        budget.facts(sum(map(len,relations.values()))+len(supports)+sum(map(len,supports.values())))
    return supports


def ground(source, n, background, arity, budget, roots=()):
    from benchmarks.ilp import parse_program, background_schema, _join
    schema = background_schema(background)
    rules = parse_program(source, schema, arity)
    edb = {p for p,_ in background} | {'true','false'}
    if roots and _acyclic(rules,edb):
        return _query_ground(rules,n,background,roots,budget)
    possible = {p:set(rows) for p,rows in background}
    possible.update(true={()}, false=set())
    supports = defaultdict(set)
    # Positive relevance closure: impossible IDB tuples need not be encoded.
    # Keep *all* surviving ground rule supports, including recursive edges;
    # candidate truth is still represented by SMT Booleans and rank proofs.
    while True:
        budget.check()
        additions = defaultdict(set)
        for (pred,args),body in rules:
            bindings = [{}]
            for atom in body:
                bindings = _join(bindings,atom,possible,budget)
            for binding in bindings:
                free = sorted(set(args)-binding.keys())
                for values in product(range(n),repeat=len(free)):
                    budget.step()
                    env = {**binding,**dict(zip(free,values))}
                    row = tuple(env[v] for v in args)
                    deps = tuple(sorted(set((p,tuple(env[v] for v in av)) for p,av in body if p not in edb)))
                    supports[(pred,row)].add(deps)
                    additions[pred].add(row)
        changed = False
        for pred,rows in additions.items():
            current = possible.setdefault(pred,set())
            if not rows<=current:
                current.update(rows); changed=True
        budget.facts(sum(len(rows) for rows in possible.values())+sum(len(v) for v in supports.values()))
        if not changed:
            return supports


def encode(solver, supports, tag, budget):
    boolean = solver.getBooleanSort(); integer=solver.getIntegerSort()
    atoms = {atom:solver.mkConst(boolean,f'{tag}_a{i}') for i,atom in enumerate(supports)}
    ranks = {atom:solver.mkConst(integer,f'{tag}_r{i}') for i,atom in enumerate(supports)}
    def conjunction(terms):
        return solver.mkBoolean(True) if not terms else terms[0] if len(terms)==1 else solver.mkTerm(K.AND,*terms)
    def disjunction(terms):
        return solver.mkBoolean(False) if not terms else terms[0] if len(terms)==1 else solver.mkTerm(K.OR,*terms)
    for head,alternatives in supports.items():
        budget.step()
        proofs=[]
        for deps in alternatives:
            budget.step()
            body=conjunction([atoms[d] for d in deps])
            solver.assertFormula(solver.mkTerm(K.IMPLIES,body,atoms[head]))
            proofs.append(conjunction([body]+[solver.mkTerm(K.GT,ranks[head],ranks[d]) for d in deps]))
        solver.assertFormula(solver.mkTerm(K.IMPLIES,atoms[head],disjunction(proofs)))
        solver.assertFormula(solver.mkTerm(K.GEQ,ranks[head],solver.mkInteger(0)))
        solver.assertFormula(solver.mkTerm(K.LEQ,ranks[head],solver.mkInteger(len(atoms))))
    return atoms


def search(source, problem, *, limit=8, timeout_ms=10_000, wall_seconds=30,
           max_work=10_000_000, max_facts=300_000):
    if limit<1 or timeout_ms<1 or wall_seconds<=0:
        raise ValueError('Invalid ILP SMT search budget')
    details={'backend':'cvc5_python_api','policy':'finite_Horn_ranked_smt_v1',
             'encoding':'ground_Horn_closure_plus_well_founded_derivation_ranks',
             'grounding':'acyclic_query_grounding_or_recursive_positive_support_closure',
             'guarantee':'finite_least_fixed_point_on_declared_contexts',
             'domain':'training_label_pool_and_all_evaluation_target_atoms',
             'unbounded_rule_equivalence_proved':False,'contexts':[],
             'timeout_ms_per_check':timeout_ms,'wall_seconds':wall_seconds,
             'max_work':max_work,'max_facts_and_supports':max_facts,
             'queries':0,'complete':False,'fallback':None}
    witnesses=[]; started=monotonic()
    budget=Budget(wall_seconds,max_work,max_facts)
    try:
        solver=cvc5.Solver(_TERM_MANAGER)
        solver.setLogic('QF_LIA')
        solver.setOption('produce-models','true');solver.setOption('incremental','true')
        checks=[]
        for tag,pairs in (('train',problem.train),('eval',problem.eval)):
            n=pairs[0][0]['num_constants']
            roots=[]
            for x,_ in pairs:
                _,value=problem._query(x)
                roots.append(('target',(value,) if problem.arity==1 else value))
            supports=ground(source,n,problem.backgrounds[n],problem.arity,budget,roots)
            # Query-directed slicing avoids encoding irrelevant auxiliary
            # atoms (e.g. every same-color pair in a 512-node graph).
            relevant=set(); todo=[]
            for x,_ in pairs:
                _,value=problem._query(x)
                args=(value,) if problem.arity==1 else value
                todo.append(('target',args))
            while todo:
                budget.step()
                head=todo.pop()
                if head in relevant or head not in supports: continue
                relevant.add(head)
                for deps in supports[head]: todo.extend(deps)
            sliced={head:alts for head,alts in supports.items() if head in relevant}
            atoms=encode(solver,sliced,tag,budget)
            details['contexts'].append({'split':tag,'num_constants':n,'target_atoms_checked':len(pairs),
                                        'ground_atoms':len(supports),'encoded_atoms':len(sliced),'ground_supports':sum(map(len,supports.values()))})
            for x,y in pairs:
                _,value=problem._query(x)
                args=(value,) if problem.arity==1 else value
                actual=atoms.get(('target',args),solver.mkBoolean(False))
                mismatch=solver.mkTerm(K.NOT,actual) if y else actual
                checks.append((mismatch,x,y))
        differences=[t for t,_,_ in checks]
        solver.assertFormula(solver.mkTerm(K.OR,*differences) if len(differences)>1 else differences[0])
        while len(witnesses)<limit:
            budget.check()
            remaining=max(1,int((budget.deadline-monotonic())*1000))
            solver.setOption('tlimit-per',str(min(timeout_ms,remaining)))
            result=solver.checkSat();details['queries']+=1
            if result.isUnsat():
                details.update(status='counterexample' if witnesses else 'finite_unsat',complete=True)
                return witnesses,False if witnesses else True,details
            if not result.isSat():
                details.update(status='unknown',reason=str(result))
                return witnesses,False if witnesses else None,details
            # Extract several distinct mismatching ground atoms per SAT model.
            batch=[]
            for mismatch,x,y in checks:
                budget.step()
                if solver.getValue(mismatch).getBooleanValue():
                    batch.append((mismatch,x,y))
                    if len(batch)>=limit-len(witnesses):break
            if not batch:
                details.update(status='translation_mismatch',reason='SAT model has no differing target atom')
                return witnesses,False if witnesses else None,details
            for mismatch,x,y in batch:
                budget.check()
                # Execution is used only to validate solver witnesses, never
                # as a fallback verdict when SMT is unknown.
                with execution_deadline(budget.deadline):
                    replay_ok = problem.predict(source,x)!=y and problem.reference_output(x)==y
                if not replay_ok:
                    details.update(status='translation_mismatch',reason='SMT counterexample did not replay')
                    return witnesses,False if witnesses else None,details
                witnesses.append((x,y))
                solver.assertFormula(solver.mkTerm(K.NOT,mismatch))
        details['status']='counterexample'
        return witnesses,False,details
    except ResourceLimit as exc:
        details.update(status='resource_limit',reason=str(exc))
        return witnesses,False if witnesses else None,details
    finally:
        details['elapsed_seconds']=round(monotonic()-started,6)
