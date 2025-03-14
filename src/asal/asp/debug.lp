#script (python)
import clingo
N = clingo.Number
S = clingo.String

def precision(tps, fps):
    p = float(tps) / (tps + fps) if tps + fps > 0 else 0.0
    return p

def recall(tps, fns):
    r = float(tps) / (tps + fns) if tps + fns > 0 else 0.0
    return r

def f1(tps, fps, fns):
    _tps, _fps, _fns = tps.number, fps.number, fns.number
    p = precision(_tps, _fps)
    r = recall(_tps, _fns)
    f1 = 2 * p * r / (p + r) if p + r > 0.0 else 0.0
    print('Hey!', f1)
    s = str(f1)
    return S(s) 
#end.


#include "domain.lp".
#include "interpreter.lp".
#include "../../data/mnist/folds/fold_0/train.csv".

falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).
falsePositive(SeqId) :- negative(SeqId), accepted(SeqId).
truePositive(SeqId) :- positive(SeqId), accepted(SeqId).

targetClass(1).

positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).
falsePositive(SeqId) :- negative(SeqId), accepted(SeqId).
truePositive(SeqId) :- positive(SeqId), accepted(SeqId).

fns(X) :- X = #count{S: falseNegative(S)}.
fps(X) :- X = #count{S: falsePositive(S)}.
tps(X) :- X = #count{S: truePositive(S)}.

f1(F) :- tps(X), fps(Y), fns(Z), F = @f1(X,Y,Z).

accepting(4). transition(1,f(1,4),4). transition(2,f(2,2),2). transition(3,f(3,2),2). 
transition(4,f(4,4),4). transition(1,f(1,1),1). transition(3,f(3,3),3). transition(2,f(2,3),3).

holds(f(1,4),S,T) :- holds(equals(even_gt_6,1),S,T).
holds(f(2,2),S,T) :- holds(equals(even_gt_6,1),S,T).
holds(f(3,2),S,T) :- holds(equals(even_gt_6,1),S,T).
holds(f(4,4),S,T) :- holds(equals(even_gt_6,1),S,T).   

holds(f(1,1),S,T) :- holds(equals(even_leq_6_gt_3,1),S,T).
holds(f(2,3),S,T) :- holds(equals(even_leq_6_gt_3,1),S,T).
holds(f(3,3),S,T) :- holds(equals(even_leq_6_gt_3,1),S,T).
holds(f(4,4),S,T) :- holds(equals(even_leq_6_gt_3,1),S,T). 

holds(f(1,1),S,T) :- holds(equals(even_leq_3,1),S,T).
holds(f(2,2),S,T) :- holds(equals(even_leq_3,1),S,T).
holds(f(3,3),S,T) :- holds(equals(even_leq_3,1),S,T).
holds(f(4,4),S,T) :- holds(equals(even_leq_3,1),S,T).

holds(f(1,1),S,T) :- holds(equals(odd_gt_6,1),S,T).
holds(f(2,3),S,T) :- holds(equals(odd_gt_6,1),S,T).
holds(f(3,3),S,T) :- holds(equals(odd_gt_6,1),S,T).
holds(f(4,4),S,T) :- holds(equals(odd_gt_6,1),S,T).

holds(f(1,1),S,T) :- holds(equals(odd_leq_6_gt_3,1),S,T).
holds(f(2,3),S,T) :- holds(equals(odd_leq_6_gt_3,1),S,T).
holds(f(3,3),S,T) :- holds(equals(odd_leq_6_gt_3,1),S,T).
holds(f(4,4),S,T) :- holds(equals(odd_leq_6_gt_3,1),S,T).

holds(f(1,1),S,T) :- holds(equals(odd_leq_3,1),S,T).
holds(f(2,3),S,T) :- holds(equals(odd_leq_3,1),S,T).
holds(f(3,2),S,T) :- holds(equals(odd_leq_3,1),S,T).
holds(f(4,4),S,T) :- holds(equals(odd_leq_3,1),S,T).

#show f1/1.
