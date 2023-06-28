#include "domain.lp".
#include "constraints.lp".
#include "template.lp".
#include "interpreter.lp".

#program base.

%------------------------------------------------------------------------------------------
% We represent states by integers. The start state is always 1. Regarding the accepting
% state, for simplicity we restrict ourselves to automata with a unique accepting state
% which equals the max_states number provided as a runtime arg. A declaration of the
% accepting state is included in template.lp
%------------------------------------------------------------------------------------------
start(1).
state(S) :- transition(S,_,_).
state(S) :- transition(_,_,S).
state(S) :- rule(f(S,_)).
state(S) :- rule(f(_,S)).

%------------------------------------------------------------------------------------------
% The head of the rule below is the body of each holds(f(I,J),S,T) predicate, which are the
% heads of the transition guards (see template.lp). In the body of the rule below we
% collect all...
%------------------------------------------------------------------------------------------
holds(body(I,J),S,T) :- rule(I), conjunction(J), sequence(S), time(T), holds(F,S,T) : body(I,J,F).

%---------------
% Generate part.
%---------------

{body(I,J,at_most(A,V)) : rule(I), conjunction(J), numerical(A), value(A,V)}.
{body(I,J,at_least(A,V)) : rule(I), conjunction(J), numerical(A), value(A,V)}.
{body(I,J,lt(A1,A2)) : rule(I), conjunction(J), numerical(A1), numerical(A2), A1 != A2}.
{body(I,J,equals(A,V)) : rule(I), conjunction(J), categorical(A), value(A,V)}.
% {body(I,J,neg(A,V)) : rule(I), conjunction(J), categorical(A), value(A,V)}.

{conjunction(1..2)}.

%--------------------------------------------
% Used to minimize used attributes.
%--------------------------------------------
used_attribute(A) :- body(_,_,at_least(A,_)).
used_attribute(A) :- body(_,_,at_most(A,_)).
used_attribute(A) :- body(_,_,equals(A,_)).
% used_attribute(A) :- body(_,_,neg(A,_)).
used_attribute(A) :- body(_,_,lt(A,_)).
used_attribute(A) :- body(_,_,lt(_,A)).

#minimize{1@2,Seq: falseNegative(Seq)}.
#minimize{1@2,Seq: falsePositive(Seq)}.
#minimize{C@2,I,J,F: body(I,J,F), cost(F,C)}.
#minimize{1@2,X: used_attribute(X)}.

targetClass(1).

positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).
falsePositive(SeqId) :- negative(SeqId), accepted(SeqId).
truePositive(SeqId) :- positive(SeqId), accepted(SeqId).

fns(X) :- X = #count{S: falseNegative(S)}.
fps(X) :- X = #count{S: falsePositive(S)}.
tps(X) :- X = #count{S: truePositive(S)}.

#show tps/1.
#show fps/1.
#show fns/1.
#show body/3.
#show transition/3.
#show accepting/1.
#show used_attribute/1.
#show rule/1.
