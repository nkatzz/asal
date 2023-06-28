#include "domain.lp".

inState(SeqId,f(1,1),1,1,1) :- sequence(SeqId).
inState(SeqId,F,S1,S2,T+1) :- inState(SeqId,_,_,S1,T), transition(S1,F,S2), holds(F,SeqId,T).
accepted(SeqId) :- inState(SeqId,_,_,S,T), accepting(S), seqEnd(SeqId,T).

seq(SeqId,T) :- seq(SeqId,_,T).
seqEnd(SeqId,T+1) :- seq(SeqId,T), not seq(SeqId,T+1).

positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

tp(SeqId) :- positive(SeqId), accepted(SeqId).
fp(SeqId) :- negative(SeqId), accepted(SeqId).
fn(SeqId) :- positive(SeqId), not accepted(SeqId).

sequence(S) :- seq(S,_,_).
step(T) :- seq(_,_,T).

% Adding the time T as output in the head helps sort by T the accepting_path_edge/4 atoms (grouped by SeqId)
% and extract the transition path by simply traversing the sorted list. Also, adding F allows to track which
% particular rule is used to move from state p to state q in a path that involves the (p,q) edge. Rules are
% properly indexed here to allow to discriminate between two alternative (i.e. disjunctive) rules with the
% same rule head.

accepting_path_edge(SeqId,S1,S2,T,F) :- inState(SeqId,F,S1,S2,T), S1!=S2, accepted(SeqId).

#show accepting_path_edge/5.
#show tp/1.
#show fp/1.
#show fn/1.
