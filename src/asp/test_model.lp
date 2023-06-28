#include "domain.lp".
#include "interpreter.lp".

targetClass(1).

positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

%*
falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).
falsePositive(SeqId) :- negative(SeqId), accepted(SeqId).
truePositive(SeqId) :- positive(SeqId), accepted(SeqId).

fns(X) :- X = #count{S: falseNegative(S)}.
fps(X) :- X = #count{S: falsePositive(S)}.
tps(X) :- X = #count{S: truePositive(S)}.
*%

tp(SeqId) :- positive(SeqId), accepted(SeqId).
fp(SeqId) :- negative(SeqId), accepted(SeqId).
fn(SeqId) :- positive(SeqId), not accepted(SeqId).

#show tp/1.
#show fp/1.
#show fn/1.
