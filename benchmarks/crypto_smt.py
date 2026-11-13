"""Bounded symbolic execution of Python fn(x) and cvc5 counterexample search.

All positions are symbolic ASCII letters, not enumerated/random words. Unsupported
Python is inconclusive. UNSAT certifies the recorded bounded domain only.
"""
from dataclasses import dataclass
import ast
import string

import cvc5
from cvc5 import Kind as K

# Keep term ownership alive across repeated queries and Python closure collection.
_TERM_MANAGER = cvc5.TermManager()


class Unsupported(ValueError):
    pass


class RuntimeFault(Exception):
    pass


@dataclass(frozen=True)
class Text:
    chars: tuple | None
    sequence: object = None
    ascii: bool = True


@dataclass
class State:
    env: dict
    pc: object
    status: str = 'running'
    value: object = None


def clone(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, tuple):
        if id(value) not in memo:
            memo[id(value)] = tuple(clone(v, memo) for v in value)
        return memo[id(value)]
    if isinstance(value, (list, dict)):
        if id(value) in memo:
            return memo[id(value)]
        result = [] if isinstance(value, list) else {}
        memo[id(value)] = result
        if isinstance(value, list):
            result.extend(clone(v, memo) for v in value)
        else:
            result.update((k, clone(v, memo)) for k, v in value.items())
        return result
    return value


class SymbolicProgram:
    def __init__(self, solver):
        self.s = solver
        self.errors = []
        self.steps = 0

    def term(self, value):
        if isinstance(value, bool): return self.s.mkBoolean(value)
        if isinstance(value, int): return self.s.mkInteger(value)
        if isinstance(value, cvc5.Term): return value
        raise Unsupported('Expected a symbolic integer or Boolean')

    def op(self, kind, *values):
        term = self.s.simplify(self.s.mkTerm(kind, *(self.term(v) for v in values)))
        if term.isBooleanValue(): return term.getBooleanValue()
        if term.isIntegerValue(): return term.getIntegerValue()
        return term

    def both(self, *values):
        if not values: return True
        if len(values) == 1: return values[0]
        return self.op(K.AND, *values)

    def either(self, *values):
        if not values: return False
        if len(values) == 1: return values[0]
        return self.op(K.OR, *values)

    def not_(self, value): return self.op(K.NOT, value)

    def text(self, value):
        if any(ord(c) > 127 for c in value):
            raise Unsupported('Only ASCII literal string operations are supported')
        return Text(tuple(ord(c) for c in value))

    def seq(self, value):
        if not isinstance(value, Text): raise Unsupported('Expected a string result')
        if value.chars is None: return value.sequence
        pieces = [self.s.mkTerm(K.SEQ_UNIT, self.term(c)) for c in value.chars]
        if not pieces: return self.s.mkEmptySequence(self.s.getIntegerSort())
        if len(pieces) == 1: return pieces[0]
        return self.s.mkTerm(K.SEQ_CONCAT, *pieces)

    def concat(self, a, b):
        if isinstance(a, Text) and isinstance(b, Text):
            if a.chars is not None and b.chars is not None:
                return Text(a.chars + b.chars, ascii=a.ascii and b.ascii)
            return Text(None, self.s.mkTerm(K.SEQ_CONCAT, self.seq(a), self.seq(b)), a.ascii and b.ascii)
        if isinstance(a, list) and isinstance(b, list): return a + b
        return self.op(K.ADD, a, b)

    def merge(self, condition, yes, no):
        if condition is True: return yes
        if condition is False: return no
        if isinstance(yes, Text) and isinstance(no, Text):
            if yes.chars is not None and no.chars is not None and len(yes.chars) == len(no.chars):
                return Text(tuple(self.op(K.ITE, condition, a, b) for a,b in zip(yes.chars,no.chars)),
                            ascii=yes.ascii and no.ascii)
            raise Unsupported('Conditional expression changes symbolic string length')
        if isinstance(yes, list) and isinstance(no, list) and len(yes) == len(no):
            return [self.merge(condition,a,b) for a,b in zip(yes,no)]
        if isinstance(yes, dict) and isinstance(no, dict) and yes.keys() == no.keys():
            return {key:self.merge(condition,yes[key],no[key]) for key in yes}
        if yes is None and no is None: return None
        if isinstance(yes, (int,bool,cvc5.Term)) and isinstance(no,(int,bool,cvc5.Term)):
            return self.op(K.ITE,condition,yes,no)
        if isinstance(yes,str) and isinstance(no,str) and yes==no: return yes
        raise Unsupported('Cannot merge these Python values across symbolic branches')

    def truth(self, value):
        if isinstance(value, Text):
            return bool(value.chars) if value.chars is not None else self.op(K.GT,self.s.mkTerm(K.SEQ_LENGTH,self.seq(value)),0)
        if isinstance(value,(list,dict,tuple,range)): return bool(value)
        if value is None: return False
        if isinstance(value,cvc5.Term) and value.getSort().isBoolean(): return value
        if isinstance(value,bool): return value
        return self.op(K.DISTINCT,value,0)

    def equal(self,a,b):
        if isinstance(a,str): a=self.text(a)
        if isinstance(b,str): b=self.text(b)
        if isinstance(a,Text) and isinstance(b,Text):
            if a.chars is not None and b.chars is not None:
                if len(a.chars) != len(b.chars): return False
                return self.both(*(self.op(K.EQUAL,x,y) for x,y in zip(a.chars,b.chars)))
            return self.op(K.EQUAL,self.seq(a),self.seq(b))
        if isinstance(a,(int,bool,cvc5.Term)) and isinstance(b,(int,bool,cvc5.Term)):
            return self.op(K.EQUAL,a,b)
        if a is None or b is None: return a is None and b is None
        raise Unsupported('Unsupported equality operands')

    def require(self,state,condition):
        failure=self.both(state.pc,self.not_(condition))
        if failure is not False: self.errors.append(failure)
        state.pc=self.both(state.pc,condition)
        if state.pc is False: raise RuntimeFault()

    def iterable(self,value):
        if isinstance(value,Text) and value.chars is not None:
            return [Text((c,),ascii=value.ascii) for c in value.chars]
        if isinstance(value,(list,tuple,range)):
            if len(value)>256: raise Unsupported('Loop exceeds bounded unrolling limit')
            return list(value)
        raise Unsupported('Iteration requires a statically bounded sequence')

    def key(self,value):
        if isinstance(value,Text) and value.chars is not None and all(isinstance(c,int) for c in value.chars):
            return ''.join(chr(c) for c in value.chars)
        if isinstance(value,(str,int,bool)): return value
        raise Unsupported('Dictionary construction requires concrete keys')

    def assign(self,target,value,state):
        if isinstance(target,ast.Name): state.env[target.id]=value; return
        if isinstance(target,(ast.Tuple,ast.List)):
            values=self.iterable(value)
            if len(values)!=len(target.elts): self.require(state,False)
            for t,v in zip(target.elts,values): self.assign(t,v,state)
            return
        if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name):
            container=state.env[target.value.id]; index=self.expr(target.slice,state)
            if isinstance(container,list) and isinstance(index,int):
                self.require(state,-len(container)<=index<len(container));container[index]=value;return
            if isinstance(container,dict):container[self.key(index)]=value;return
        raise Unsupported('Unsupported assignment target')

    def choice(self,node,condition,state,yes,no):
        if isinstance(condition,bool): return yes(state) if condition else no(state)
        a=State(clone(state.env),self.both(state.pc,condition))
        b=State(clone(state.env),self.both(state.pc,self.not_(condition)))
        va=vb=None
        try: va=yes(a)
        except RuntimeFault: a.pc=False
        try: vb=no(b)
        except RuntimeFault: b.pc=False
        state.pc=self.either(a.pc,b.pc)
        if state.pc is False: raise RuntimeFault()
        if a.pc is False:return vb
        if b.pc is False:return va
        return self.merge(condition,va,vb)

    def expr(self,node,state):
        self.steps+=1
        if self.steps>50000: raise Unsupported('Symbolic translation instruction budget exceeded')
        if isinstance(node,ast.Constant):
            if isinstance(node.value,str):return self.text(node.value)
            if node.value is None or isinstance(node.value,(int,bool)):return node.value
            raise Unsupported('Unsupported literal')
        if isinstance(node,ast.Name):
            if node.id in state.env:return state.env[node.id]
            if node.id in ('str','string'):return node.id
            raise Unsupported('Unbound or unsupported name: '+node.id)
        if isinstance(node,ast.Attribute):
            base=self.expr(node.value,state)
            if isinstance(base,str) and base=='string' and node.attr in ('ascii_lowercase','ascii_uppercase','ascii_letters'):
                return self.text(getattr(string,node.attr))
            raise Unsupported('Unsupported attribute')
        if isinstance(node,(ast.List,ast.Tuple)):
            values=[self.expr(v,state) for v in node.elts]
            return tuple(values) if isinstance(node,ast.Tuple) else values
        if isinstance(node,ast.Dict):
            return {self.key(self.expr(k,state)):self.expr(v,state) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.IfExp):
            condition=self.truth(self.expr(node.test,state))
            return self.choice(node,condition,state,lambda st:self.expr(node.body,st),lambda st:self.expr(node.orelse,st))
        if isinstance(node,ast.UnaryOp):
            value=self.expr(node.operand,state)
            if isinstance(node.op,ast.Not):return self.not_(self.truth(value))
            if isinstance(node.op,ast.USub):return self.op(K.NEG,value)
            if isinstance(node.op,ast.UAdd):return value
        if isinstance(node,ast.BinOp):
            a,b=self.expr(node.left,state),self.expr(node.right,state)
            if isinstance(node.op,ast.Add):return self.concat(a,b)
            if isinstance(node.op,ast.Sub):return self.op(K.SUB,a,b)
            if isinstance(node.op,ast.Mult):
                if isinstance(a,(list,Text)) or isinstance(b,(list,Text)):
                    sequence,count=(a,b) if isinstance(a,(list,Text)) else (b,a)
                    if not isinstance(count,int) or abs(count)>256:raise Unsupported('Unbounded repetition')
                    result=[] if isinstance(sequence,list) else self.text('')
                    for _ in range(max(count,0)):result=self.concat(result,sequence)
                    return result
                return self.op(K.MULT,a,b)
            if isinstance(node.op,(ast.Mod,ast.FloorDiv)):
                if not isinstance(b,int):raise Unsupported('Only constant divisors are supported')
                if b==0:self.require(state,False)
                if b<0:raise Unsupported('Negative divisors are not translated')
                return self.op(K.INTS_MODULUS if isinstance(node.op,ast.Mod) else K.INTS_DIVISION,a,b)
        if isinstance(node,ast.BoolOp):
            value=self.expr(node.values[0],state)
            for next_node in node.values[1:]:
                condition=self.truth(value);previous=value
                if isinstance(node.op,ast.And):
                    value=self.choice(node,condition,state,lambda st:self.expr(next_node,st),lambda st:previous)
                else:value=self.choice(node,condition,state,lambda st:previous,lambda st:self.expr(next_node,st))
            return value
        if isinstance(node,ast.Compare):
            if len(node.ops)>1:
                # ASCII range tests are supported; avoid eager evaluation of
                # error-producing chained comparators with Python short circuit.
                if not all(isinstance(v,(ast.Constant,ast.Name)) for v in [node.left]+node.comparators):
                    raise Unsupported('Complex chained comparison')
            a=self.expr(node.left,state);conditions=[]
            for op,rhs in zip(node.ops,node.comparators):
                b=self.expr(rhs,state)
                if isinstance(op,(ast.In,ast.NotIn)):
                    if isinstance(a,Text) and isinstance(b,Text):
                        if a.chars is None or b.chars is None: raise Unsupported('Variable-length substring test')
                        width=len(a.chars)
                        condition=self.either(*(self.equal(a,Text(b.chars[i:i+width],ascii=b.ascii)) for i in range(len(b.chars)-width+1)))
                    elif isinstance(b,dict):condition=self.either(*(self.equal(a,k) for k in b))
                    else:condition=self.either(*(self.equal(a,v) for v in self.iterable(b)))
                    if isinstance(op,ast.NotIn):condition=self.not_(condition)
                elif isinstance(op,(ast.Eq,ast.NotEq)):
                    condition=self.equal(a,b)
                    if isinstance(op,ast.NotEq):condition=self.not_(condition)
                else:
                    if isinstance(a,Text) and isinstance(b,Text):
                        if a.chars is None or b.chars is None or len(a.chars)!=1 or len(b.chars)!=1:
                            raise Unsupported('Only single-character ordering is translated')
                        left,right=a.chars[0],b.chars[0]
                    else:left,right=a,b
                    kinds={ast.Lt:K.LT,ast.LtE:K.LEQ,ast.Gt:K.GT,ast.GtE:K.GEQ}
                    if type(op) not in kinds:raise Unsupported('Unsupported comparison')
                    condition=self.op(kinds[type(op)],left,right)
                conditions.append(condition);a=b
            return self.both(*conditions)
        if isinstance(node,ast.Subscript):
            value=self.expr(node.value,state)
            if isinstance(node.slice,ast.Slice):
                if not isinstance(value,(Text,list,tuple)):raise Unsupported('Unsupported slice')
                indices=[self.expr(v,state) if v else None for v in (node.slice.lower,node.slice.upper,node.slice.step)]
                if any(i is not None and not isinstance(i,int) for i in indices):raise Unsupported('Symbolic slice bounds')
                if indices[2]==0:self.require(state,False)
                if isinstance(value,Text):
                    if value.chars is None:raise Unsupported('Slice of a variable-length symbolic result')
                    return Text(value.chars[slice(*indices)],ascii=value.ascii)
                return value[slice(*indices)]
            index=self.expr(node.slice,state)
            if isinstance(value,dict):return self.lookup(value,index,state)
            values=self.iterable(value);n=len(values)
            if isinstance(index,int):
                self.require(state,-n<=index<n);return values[index]
            self.require(state,self.both(self.op(K.GEQ,index,-n),self.op(K.LT,index,n)))
            pos=self.op(K.ITE,self.op(K.LT,index,0),self.op(K.ADD,index,n),index)
            result=values[-1]
            for i in reversed(range(n-1)):result=self.merge(self.op(K.EQUAL,pos,i),values[i],result)
            return result
        if isinstance(node,(ast.GeneratorExp,ast.ListComp,ast.DictComp)):
            values=[]
            def visit(index,st):
                if index==len(node.generators):
                    if isinstance(node,ast.DictComp):values.append((self.key(self.expr(node.key,st)),self.expr(node.value,st)))
                    else:values.append(self.expr(node.elt,st))
                    return
                generator=node.generators[index]
                for value in self.iterable(self.expr(generator.iter,st)):
                    child=State(clone(st.env),st.pc);self.assign(generator.target,value,child)
                    for test in generator.ifs:
                        flag=self.truth(self.expr(test,child))
                        if not isinstance(flag,bool):raise Unsupported('Symbolically filtered comprehension')
                        if not flag:break
                    else:visit(index+1,child)
                    st.pc=child.pc
            visit(0,state)
            return dict(values) if isinstance(node,ast.DictComp) else values
        if isinstance(node,ast.Call):return self.call(node,state)
        raise Unsupported('Unsupported expression: '+type(node).__name__)

    def lookup(self,mapping,key,state,default=None,has_default=False):
        try:concrete=self.key(key)
        except Unsupported:concrete=None
        if concrete is not None:
            if concrete in mapping:return mapping[concrete]
            if has_default:return default
            self.require(state,False)
        matches=[(self.equal(key,k),v) for k,v in mapping.items()]
        if not has_default:self.require(state,self.either(*(cond for cond,_ in matches)))
        if not matches:
            if has_default:return default
            self.require(state,False)
        result=default if has_default else matches[-1][1]
        for condition,value in reversed(matches):result=self.merge(condition,value,result)
        return result

    def call(self,node,state):
        if node.keywords:raise Unsupported('Keyword arguments are not translated')
        args=[self.expr(a,state) for a in node.args]
        if isinstance(node.func,ast.Name):
            name=node.func.id
            if name in state.env and not (name=='str' and state.env[name]=='str'):
                raise Unsupported('Calling a shadowed builtin')
            if name=='ord' and len(args)==1:
                self.require(state,isinstance(args[0],Text) and args[0].chars is not None and len(args[0].chars)==1)
                return args[0].chars[0]
            if name=='chr' and len(args)==1:
                self.require(state,self.both(self.op(K.GEQ,args[0],0),self.op(K.LEQ,args[0],0x10ffff)))
                return Text((args[0],),ascii=isinstance(args[0],int) and 0<=args[0]<128)
            if name=='len' and len(args)==1:
                if isinstance(args[0],Text):
                    return len(args[0].chars) if args[0].chars is not None else self.s.mkTerm(K.SEQ_LENGTH,self.seq(args[0]))
                return len(args[0])
            if name in ('list','tuple','reversed') and len(args)==1:
                values=self.iterable(args[0])
                if name=='reversed': raise Unsupported('reversed iterator is not translated')
                return tuple(values) if name=='tuple' else values
            if name=='range' and all(isinstance(a,int) for a in args):
                try:r=range(*args)
                except ValueError:self.require(state,False)
                if len(r)>256: raise Unsupported('Range exceeds unrolling budget')
                return r
            if name=='enumerate' and 1<=len(args)<=2:
                start=args[1] if len(args)==2 else 0
                if not isinstance(start,int):raise Unsupported('Symbolic enumerate start')
                return list(enumerate(self.iterable(args[0]),start))
            if name=='zip':return list(zip(*(self.iterable(a) for a in args)))
            if name=='dict' and len(args)==1:return {self.key(k):v for k,v in self.iterable(args[0])}
            if name in ('all','any') and len(args)==1:
                values=[self.truth(v) for v in self.iterable(args[0])]
                return self.both(*values) if name=='all' else self.either(*values)
            if name=='bool' and len(args)==1:return self.truth(args[0])
            if name=='int' and len(args)==1 and isinstance(args[0],(int,bool,cvc5.Term)):return args[0]
            if name=='str' and len(args)==1 and isinstance(args[0],int):return self.text(str(args[0]))
            if name=='sum' and len(args)==1:
                result=0
                for v in self.iterable(args[0]):result=self.op(K.ADD,result,v)
                return result
            raise Unsupported('Unsupported function: '+name)
        if not isinstance(node.func,ast.Attribute):raise Unsupported('Unsupported callable')
        obj=self.expr(node.func.value,state);name=node.func.attr
        if name=='maketrans' and obj=='str' and len(args)==2:
            a,b=self.iterable(args[0]),self.iterable(args[1]);self.require(state,len(a)==len(b))
            return {ord(self.key(x)):ord(self.key(y)) for x,y in zip(a,b)}
        if isinstance(obj,dict) and name=='get' and 1<=len(args)<=2:
            return self.lookup(obj,args[0],state,args[1] if len(args)==2 else None,True)
        if not isinstance(obj,Text):raise Unsupported('Unsupported method: '+name)
        if name=='join' and len(args)==1:
            result=self.text('')
            for i,value in enumerate(self.iterable(args[0])):
                if i:result=self.concat(result,obj)
                if not isinstance(value,Text):self.require(state,False)
                result=self.concat(result,value)
            return result
        if name in ('upper','lower','isupper','islower','isalpha') and not args:
            if obj.chars is None or not obj.ascii:raise Unsupported('Case operations require known ASCII strings')
            uppers=[self.both(self.op(K.GEQ,c,65),self.op(K.LEQ,c,90)) for c in obj.chars]
            lowers=[self.both(self.op(K.GEQ,c,97),self.op(K.LEQ,c,122)) for c in obj.chars]
            if name=='isalpha':return self.both(bool(obj.chars),*(self.either(u,l) for u,l in zip(uppers,lowers)))
            if name=='isupper':return self.both(self.either(*uppers),*(self.not_(l) for l in lowers))
            if name=='islower':return self.both(self.either(*lowers),*(self.not_(u) for u in uppers))
            flags=lowers if name=='upper' else uppers;delta=-32 if name=='upper' else 32
            return Text(tuple(self.op(K.ITE,flag,self.op(K.ADD,c,delta),c) for flag,c in zip(flags,obj.chars)))
        if name in ('index','find') and len(args)==1:
            values=self.iterable(obj);needle=args[0]
            if not isinstance(needle,Text) or needle.chars is None or len(needle.chars)!=1:
                raise Unsupported('Only single-character string search is translated')
            matches=[self.equal(v,needle) for v in values]
            if name=='index':self.require(state,self.either(*matches))
            result=-1
            for i in reversed(range(len(values))):result=self.op(K.ITE,matches[i],i,result)
            return result
        if name=='translate' and len(args)==1 and isinstance(args[0],dict):
            result=self.text('');mapping=args[0]
            for char in self.iterable(obj):
                replacement=char
                for key,value in mapping.items():
                    if not isinstance(key,int):raise Unsupported('Translation keys must be code points')
                    if value is None:converted=self.text('')
                    elif isinstance(value,int):converted=Text((value,),ascii=0<=value<128)
                    elif isinstance(value,Text):converted=value
                    else:raise Unsupported('Unsupported translation entry')
                    replacement=self.merge(self.op(K.EQUAL,char.chars[0],key),converted,replacement)
                result=self.concat(result,replacement)
            return result
        raise Unsupported('Unsupported string method: '+name)

    def stmt(self,node,state):
        if isinstance(node,ast.Return):
            state.value=self.expr(node.value,state) if node.value else None;state.status='return';return [state]
        if isinstance(node,ast.Assign):
            value=self.expr(node.value,state)
            for target in node.targets:self.assign(target,value,state)
            return [state]
        if isinstance(node,ast.AugAssign):
            old=self.expr(node.target,state);value=self.expr(node.value,state)
            if isinstance(node.op,ast.Add):new=self.concat(old,value)
            elif isinstance(node.op,ast.Sub):new=self.op(K.SUB,old,value)
            else:raise Unsupported('Unsupported augmented assignment')
            # += on a list is in-place in Python (aliases must see the update).
            if isinstance(old,list) and isinstance(node.op,ast.Add):old.extend(value)
            else:self.assign(node.target,new,state)
            return [state]
        if isinstance(node,ast.Expr):
            if isinstance(node.value,ast.Constant):return [state]  # docstring
            call=node.value
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr in ('append','extend'):
                obj=self.expr(call.func.value,state)
                if not isinstance(obj,list) or len(call.args)!=1:raise Unsupported('Expected list append/extend')
                value=self.expr(call.args[0],state)
                if call.func.attr=='append':obj.append(value)
                else:obj.extend(self.iterable(value))
                return [state]
            self.expr(node.value,state);return [state]
        if isinstance(node,ast.If):
            condition=self.truth(self.expr(node.test,state))
            if isinstance(condition,bool):return self.block(node.body if condition else node.orelse,[state])
            yes=self.block(node.body,[State(clone(state.env),self.both(state.pc,condition))])
            no=self.block(node.orelse,[State(clone(state.env),self.both(state.pc,self.not_(condition)))])
            if len(yes)==len(no)==1 and yes[0].status==no[0].status:
                a,b=yes[0],no[0]
                # Avoid losing Python alias relationships at a branch merge.
                def aliases(env):
                    seen=set()
                    def visit(value):
                        if isinstance(value,(dict,list)):
                            if id(value) in seen:return True
                            seen.add(id(value))
                            return any(visit(v) for v in (value.values() if isinstance(value,dict) else value))
                        if isinstance(value,tuple):return any(visit(v) for v in value)
                        return False
                    return any(visit(v) for v in env.values())
                if a.env.keys()==b.env.keys() and not aliases(a.env) and not aliases(b.env):
                    try:
                        env={k:self.merge(condition,a.env[k],b.env[k]) for k in a.env}
                        value=self.merge(condition,a.value,b.value)
                        return [State(env,self.either(a.pc,b.pc),a.status,value)]
                    except Unsupported:pass
            return yes+no
        if isinstance(node,ast.For):
            container=self.expr(node.iter,state)
            if isinstance(container,list):
                aliases={name for name,value in state.env.items() if value is container}
                for part in ast.walk(node):
                    if isinstance(part,(ast.Assign,ast.AugAssign)) or (isinstance(part,ast.Call) and isinstance(part.func,ast.Attribute) and part.func.attr in ('append','extend')):
                        if any(isinstance(v,ast.Name) and v.id in aliases for v in ast.walk(part)):
                            raise Unsupported('Mutation or aliasing of an iterated list')
                if any(isinstance(v,(list,dict)) for v in container):raise Unsupported('Loop over mutable elements')
            values=self.iterable(container);states=[state];finished=[]
            for value in values:
                next_states=[]
                for st in states:
                    if st.status=='return':finished.append(st);continue
                    self.assign(node.target,value,st)
                    for child in self.block(node.body,[st]):
                        if child.status=='break':child.status='running';finished.append(child)
                        else:
                            if child.status=='continue':child.status='running'
                            next_states.append(child)
                states=next_states
            return finished+self.block(node.orelse,states)
        if isinstance(node,(ast.Break,ast.Continue)):
            state.status='break' if isinstance(node,ast.Break) else 'continue';return [state]
        if isinstance(node,ast.Pass):return [state]
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            if isinstance(node,ast.Import):
                if any(n.name!='string' for n in node.names):raise Unsupported('Unsupported import')
                for n in node.names:state.env[n.asname or n.name]='string'
            else:
                if node.module!='string' or node.level:raise Unsupported('Unsupported import')
                for n in node.names:
                    if n.name not in ('ascii_lowercase','ascii_uppercase','ascii_letters'):raise Unsupported('Unsupported string constant')
                    state.env[n.asname or n.name]=self.text(getattr(string,n.name))
            return [state]
        raise Unsupported('Unsupported statement: '+type(node).__name__)

    def block(self,nodes,states):
        for node in nodes:
            updated=[]
            for state in states:
                if state.pc is False:continue
                if state.status!='running':updated.append(state);continue
                try:updated.extend(self.stmt(node,state))
                except RuntimeFault:pass
            states=updated
            if len(states)>1024:raise Unsupported('Symbolic path budget exceeded')
        return states

    def translate(self,source,chars):
        tree=ast.parse(source)
        parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node,ast.GeneratorExp):
                parent=parents.get(node)
                if not (isinstance(parent,ast.Call) and isinstance(parent.func,ast.Attribute) and parent.func.attr=='join'):
                    raise Unsupported('Generators are supported only directly inside join')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('enumerate','zip'):
                parent=parents.get(node)
                if not (isinstance(parent,(ast.For,ast.comprehension)) and parent.iter is node):
                    raise Unsupported('Iterator must be consumed directly by a loop')
        fn=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fn'),None)
        if fn is None:raise Unsupported('Expected fn(x)')
        state=State({'str':'str'},True)
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        states=self.block(imports,[state])
        state=states[0];state.env['x']=Text(tuple(chars))
        states=self.block(fn.body,[state])
        results=[]
        for st in states:
            if st.status=='return' and isinstance(st.value,Text):results.append((st.pc,st.value))
            else:self.errors.append(st.pc)  # implicit None/non-string return
        return results


def reference_char(engine,c,spec):
    upper=engine.both(engine.op(K.GEQ,c,65),engine.op(K.LEQ,c,90))
    if spec['cipher']=='caesar':
        base=engine.op(K.ITE,upper,65,97)
        return engine.op(K.ADD,engine.op(K.INTS_MODULUS,engine.op(K.ADD,engine.op(K.SUB,c,base),int(spec['shift'])),26),base)
    if spec['cipher']=='atbash':return engine.op(K.SUB,engine.op(K.ITE,upper,155,219),c)
    if spec['cipher']=='keyboard':
        keys='qwertyuiopasdfghjklzxcvbnm'
        table={ord(ch):ord(keys[i]) for i,ch in enumerate(string.ascii_lowercase)}
        table.update({ord(ch):ord(keys[i].upper()) for i,ch in enumerate(string.ascii_uppercase)})
        result=c
        for key,value in table.items():result=engine.op(K.ITE,engine.op(K.EQUAL,c,key),value,result)
        return result
    raise ValueError('Unknown reference cipher')


def search(source,spec,*,limit=8,max_length=10,timeout_ms=10000,replay=None):
    """Return (witness strings, verdict, metadata) using only symbolic search."""
    from benchmarks.crypto import compile_program
    details={'policy':'bounded_symbolic_python_crypto_v1','backend':'cvc5_python_api',
             'guarantee':'equivalent_on_bounded_domain_only',
             'domain':{'alphabet':'ASCII A-Z/a-z','min_length':0,'max_length':max_length},
             'timeout_ms_per_check':timeout_ms,'queries':0,'lengths_proved':[],
             'lengths_searched':[],'fallback':None}
    if limit<1 or max_length<0 or timeout_ms<1:raise ValueError('Invalid crypto SMT search bounds')
    compile_program(source)  # preserve the task's execution allowlist
    witnesses=[]
    for length in range(max_length+1):
        solver=cvc5.Solver(_TERM_MANAGER);solver.setLogic('ALL')
        solver.setOption('produce-models','true');solver.setOption('incremental','true')
        solver.setOption('tlimit-per',str(timeout_ms))
        engine=SymbolicProgram(solver)
        chars=[solver.mkConst(solver.getIntegerSort(),f'c{i}') for i in range(length)]
        for char in chars:
            solver.assertFormula(engine.term(engine.either(
                engine.both(engine.op(K.GEQ,char,65),engine.op(K.LEQ,char,90)),
                engine.both(engine.op(K.GEQ,char,97),engine.op(K.LEQ,char,122)))))
        try:
            outputs=engine.translate(source,chars)
            expected=Text(tuple(reference_char(engine,c,spec) for c in chars))
            mismatch=engine.either(*engine.errors,*(engine.both(pc,engine.not_(engine.equal(value,expected))) for pc,value in outputs))
            solver.assertFormula(engine.term(mismatch))
        except Unsupported as exc:
            details.update(status='unsupported',reason=str(exc))
            return witnesses,False if witnesses else None,details
        details['lengths_searched'].append(length)
        while len(witnesses)<limit:
            result=solver.checkSat();details['queries']+=1
            if result.isUnsat():
                if not witnesses:details['lengths_proved'].append(length)
                break
            if not result.isSat():
                details.update(status='unknown',reason=str(result))
                return witnesses,False if witnesses else None,details
            word=''.join(chr(solver.getValue(c).getIntegerValue()) for c in chars)
            if replay is not None and not replay(word):
                details.update(status='translation_mismatch',reason='SMT witness did not reproduce in Python')
                return witnesses,False if witnesses else None,details
            witnesses.append(word)
            if not chars:break
            solver.assertFormula(engine.term(engine.either(*(engine.op(K.DISTINCT,c,ord(v)) for c,v in zip(chars,word)))))
        if len(witnesses)>=limit:
            details['status']='counterexample';return witnesses,False,details
    details['status']='counterexample' if witnesses else 'bounded_unsat'
    return witnesses,not witnesses,details
