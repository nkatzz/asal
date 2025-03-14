#include "domain.lp".
#include "interpreter.lp".

positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

tp(SeqId) :- positive(SeqId), accepted(SeqId).
fp(SeqId) :- negative(SeqId), accepted(SeqId).
fn(SeqId) :- positive(SeqId), not accepted(SeqId).

#show tp/1.
#show fp/1.
#show fn/1.
