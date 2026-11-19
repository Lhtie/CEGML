"""Reproducible CPU execution of the pinned author's synthetic generators.

Only selected trusted class/function definitions are compiled, avoiding the
unrelated GQA/WN dependencies of the author's all-task Dataset module.
"""
import ast
from functools import lru_cache
from pathlib import Path
import importlib.util
import random

ROOT=Path(__file__).resolve().parents[1]/'datasets/ilp/source/HRI/utils'
CONFIGS={
 'adj_to_red':('AdjToRedDataset',('red','edge','color'),1),
 'two_children':('TwoChildrenDataset',('neq','edge'),1),
 'relatedness':('RelatednessDataset',('parent',),2),
 'member':('MemberDataset',('const','value'),2),
 'son':('SonDataset',('father','brother','sister'),2),
 'even_odd':('EvenOddDataset',('zero','succ'),1),
 'grandparent_nlm':('GrandparentNLMDataset',('father','mother','son','daughter'),2),
}


@lru_cache(maxsize=1)
def generators():
 import numpy as np
 import torch
 modules={}
 for name in ('Graph','Family'):
  spec=importlib.util.spec_from_file_location('_ilp_author_'+name,ROOT/(name+'.py'))
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  if name=='Family':
   # NumPy choice rejects the author's heterogeneous [None, (man,woman),...].
   # Uniformly choosing an index preserves the intended sampling operation.
   class SequenceRandom:
    def __getattr__(self,key):return getattr(np.random,key)
    def choice(self,values,*args,**kwargs):
     if isinstance(values,list) and any(v is None or isinstance(v,tuple) for v in values):
      return values[int(np.random.randint(len(values)))]
     return np.random.choice(values,*args,**kwargs)
   mod.random=SequenceRandom()
  modules[name]=mod
 ns={'np':np,'torch':torch,'random':np.random,'rnd':random,'Variable':lambda value:value,
     'product':__import__('itertools').product,
     'get_random_graph_generator':modules['Graph'].get_random_graph_generator,
     'Family':modules['Family'].Family,'randomly_generate_family':modules['Family'].randomly_generate_family}
 tree=ast.parse((ROOT/'Dataset.py').read_text())
 classes={v[0] for v in CONFIGS.values()}
 body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in classes or isinstance(n,ast.FunctionDef) and n.name=='gen_graph']
 exec(compile(ast.Module(body=body,type_ignores=[]),str(ROOT/'Dataset.py'),'exec'),ns)
 return ns


def split(ident,n,seed,limit=None):
 import numpy as np
 import torch
 cls,predicates,arity=CONFIGS[ident]
 old_np=np.random.get_state();old_py=random.getstate()
 try:
  np.random.seed(seed);random.seed(seed)
  valuations,target=generators()[cls]().getData(n)
 finally:
  np.random.set_state(old_np);random.setstate(old_py)
 background={}
 unary={'zero','red'}
 for pred,val in zip(predicates,valuations):
  facts=torch.nonzero(val.reshape(-1)).tolist() if pred in unary else torch.nonzero(val).tolist()
  background[pred]=facts
 from itertools import product
 atoms=list(product(range(n),repeat=arity));rng=random.Random(seed);rng.shuffle(atoms)
 def label(args):return bool(target.reshape(-1)[args[0]]) if arity==1 else bool(target[args])
 pos=[t for t in atoms if label(t)];neg=[t for t in atoms if not label(t)]
 ordered=[]
 for i in range(max(len(pos),len(neg))):
  if i<len(pos):ordered.append(pos[i])
  if i<len(neg):ordered.append(neg[i])
 if limit:ordered=ordered[:limit]
 return {'num_constants':n,'inference_steps':None,'background':background,
         'examples':[{'input':{'num_constants':n,'atom':'target('+','.join(map(str,t))+')'},'output':label(t)} for t in ordered]}
