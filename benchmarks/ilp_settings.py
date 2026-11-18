"""Deterministic scaleup instances of HRI synthetic target relations.

Target definitions follow the vendored Dataset.py; graph/family sampling is
our seeded sparse construction, not the author's random generator.
"""
from itertools import product
import random

SPECS = {
    'graph_coloring': ('GraphColoring', 1, 'The node has an outgoing neighbor with the same color.'),
    'length': ('Length', 2, 'The second constant is the length of the list starting at the first.'),
    'even_succ': ('EvenSucc', 1, 'Accept even-index constants.'),
    'predecessor': ('Predecessor', 2, 'The first constant immediately succeeds the second.'),
    'less_than': ('LessThan', 2, 'The first constant is smaller than the second.'),
    'fizz': ('Fizz', 1, 'Accept multiples of three, including zero.'),
    'buzz': ('Buzz', 1, 'Accept multiples of five, including zero.'),
    'connectedness': ('Connectedness', 2, 'A nonempty directed path connects the first node to the second.'),
    'cyclic': ('Cyclic', 1, 'The node lies on a directed cycle.'),
    'undirected_edge': ('UndirectedEdge', 2, 'An edge exists in either direction between the nodes.'),
    'grandparent': ('Grandparent', 2, 'The first person is a grandparent of the second.'),
    'has_father': ('HasFather', 1, 'The person has a father.'),
}
# A priori structural tiers over all 19 author synthetic configurations.
# This is a reference-rule proxy, not measured LLM difficulty. Overlapping
# author variants remain in the inventory, but are not selected twice.
AUTHOR_DIFFICULTY_TIERS = {
    'easy': ('HasFather', 'Predecessor', 'UndirectedEdge'),
    'medium': ('AdjToRed', 'Grandparent', 'Grandparent_NLM', 'Son', 'TwoChildren', 'GraphColoring'),
    'hard': ('LessThan', 'Connectedness', 'Member', 'EvenOdd', 'EvenSucc', 'Buzz', 'Cyclic', 'Relatedness', 'Fizz', 'Length'),
}
BALANCED_IDS = ('predecessor', 'undirected_edge', 'grandparent', 'graph_coloring', 'fizz', 'length')

REFERENCES = {
    'graph_coloring': 'same(X,Y) :- color(X,Z), color(Y,Z).\ntarget(X) :- edge(X,Y), same(X,Y).',
    'length': 'target(X,X) :- zero(X).\naux(X,Y) :- target(X,Z), succ(Z,Y).\ntarget(X,Y) :- const(X,Z), aux(Z,Y).',
    'predecessor': 'target(X,Y) :- succ(Y,X).',
    'less_than': 'target(X,Y) :- succ(X,Y).\ntarget(X,Y) :- succ(X,Z), target(Z,Y).',
    'fizz': 'step2(X,Y) :- succ(X,Z), succ(Z,Y).\nstep3(X,Y) :- step2(X,Z), succ(Z,Y).\ntarget(X) :- zero(X).\ntarget(Y) :- target(X), step3(X,Y).',
    'buzz': 'step5(X,Y) :- pred1(X,Z), pred3(Z,Y).\ntarget(X) :- zero(X).\ntarget(Y) :- target(X), step5(X,Y).',
    'connectedness': 'target(X,Y) :- edge(X,Y).\ntarget(X,Y) :- edge(X,Z), target(Z,Y).',
    'cyclic': 'path(X,Y) :- edge(X,Y).\npath(X,Y) :- edge(X,Z), path(Z,Y).\ntarget(X) :- path(X,X).',
    'undirected_edge': 'target(X,Y) :- edge(X,Y).\ntarget(X,Y) :- edge(Y,X).',
    'grandparent': 'parent(X,Y) :- father(X,Y).\nparent(X,Y) :- mother(X,Y).\ntarget(X,Y) :- parent(X,Z), parent(Z,Y).',
    'has_father': 'target(X) :- father(Y,X).',
}


def closure(edges, n):
    # Independent graph reachability ground truth, not the Horn interpreter.
    adjacency = {i: set() for i in range(n)}
    for a, b in edges:
        adjacency[a].add(b)
    result = set()
    for start in range(n):
        todo = list(adjacency[start]); seen = set()
        while todo:
            node = todo.pop()
            if node not in seen:
                seen.add(node); todo.extend(adjacency[node] - seen)
        result.update((start, end) for end in seen)
    return result


def make_split(ident, n, seed, limit=None):
    arity = SPECS[ident][1]
    rng = random.Random(seed)
    if ident in {'predecessor', 'less_than', 'fizz', 'buzz'}:
        bg = {'zero': [[0]], 'succ': [[i,i+1] for i in range(n-1)]}
        if ident == 'buzz':
            bg.update(pred1=[[i,i+2] for i in range(n-2)], pred3=[[i,i+3] for i in range(n-3)])
        def label(t):
            if ident == 'predecessor': return t[0] == t[1]+1
            if ident == 'less_than': return t[0] < t[1]
            return t[0] % (3 if ident == 'fizz' else 5) == 0
    elif ident == 'length':
        nodes = list(range(1,n)); rng.shuffle(nodes); nodes.append(0)
        bg = {'zero':[[0]], 'succ':[[i,i+1] for i in range(n-1)],
              'const':[[a,b] for a,b in zip(nodes,nodes[1:])]}
        lengths = {node:n-1-i for i,node in enumerate(nodes)}
        def label(t): return lengths[t[0]] == t[1]
    elif ident in {'connectedness', 'cyclic', 'undirected_edge', 'graph_coloring'}:
        # Disconnected small blocks retain both positive and negative labels,
        # including cyclic and acyclic nodes, without huge dense closures.
        edges = set()
        for base in range(0,n,8):
            nodes = list(range(base,min(base+8,n)))
            for a,b in product(nodes,repeat=2):
                if a != b and rng.random() < .16: edges.add((a,b))
            if len(nodes)>=5:
                edges.update([(nodes[0],nodes[1]),(nodes[1],nodes[2]),(nodes[2],nodes[0])])
                edges = {(a,b) for a,b in edges if a != nodes[-1] and b != nodes[-1]}
        bg = {'edge': [list(t) for t in sorted(edges)]}
        reach = closure(edges,n)
        colors = [rng.randrange(2) for _ in range(n)]
        if ident == 'graph_coloring':
            bg['color'] = [[i,c] for i,c in enumerate(colors)]
        def label(t):
            if ident == 'graph_coloring': return any(a==t[0] and colors[a]==colors[b] for a,b in edges)
            if ident == 'connectedness': return t in reach
            if ident == 'cyclic': return (t[0],t[0]) in reach
            return t in edges or t[::-1] in edges
    else:
        father, mother = set(), set()
        for base in range(0,n,8):
            # Three generations with missing parent links and isolated people.
            for a,b in [(0,2),(0,3),(2,5),(3,6)]:
                if base+b<n and rng.random()<.85: father.add((base+a,base+b))
            for a,b in [(1,2),(1,3),(4,5),(4,6)]:
                if base+b<n and rng.random()<.85: mother.add((base+a,base+b))
        bg = {'father': [list(t) for t in sorted(father)], 'mother': [list(t) for t in sorted(mother)]}
        if ident == 'has_father':
            # Additional paper background predicates; child -> parent.
            bg.update(son=[list(t[::-1]) for t in sorted(father|mother) if t[1]%2==0],
                      daughter=[list(t[::-1]) for t in sorted(father|mother) if t[1]%2==1])
        parent = father|mother
        grandparents = {(a,c) for a,b in parent for x,c in parent if b==x}
        def label(t):
            return t in grandparents if ident=='grandparent' else any(b==t[0] for a,b in father)
    atoms = list(product(range(n),repeat=arity))
    rng.shuffle(atoms)
    # Start with both labels and retain sparse relational positives in every
    # geometric prefix. All remaining atoms still appear in full evaluation.
    positive = [t for t in atoms if label(t)]
    negative = [t for t in atoms if not label(t)]
    ordered = []
    for i in range(max(len(positive),len(negative))):
        if i<len(positive): ordered.append(positive[i])
        if i<len(negative): ordered.append(negative[i])
    if limit: ordered = ordered[:limit]
    return {'num_constants':n, 'inference_steps':None, 'background':bg,
            'examples':[{'input':{'num_constants':n,'atom':'target('+','.join(map(str,t))+')'},
                         'output':bool(label(t))} for t in ordered]}
