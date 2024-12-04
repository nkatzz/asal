#include "domain.lp".
#include "constraints.lp".
#include "template.lp".
#include "interpreter.lp".

#program class(n).
targetClass(n).

#program base.

%------------------------------------------------------------------------------------------
% We represent states by integers. The start state is always 1. Regarding the accepting
% state, for simplicity we restrict ourselves to automata with a unique accepting state
% which equals the max_states number provided as a runtime arg. A declaration of the
% accepting state is included in template.lp, which is generated at runtime.
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
% {body(I,J,equals(A,V)) : rule(I), conjunction(J), categorical(A), value(A,V)}.
% {body(I,J,neg(A,V)) : rule(I), conjunction(J), categorical(A), value(A,V)}.

% {body(I,J,increase(A)) : rule(I), conjunction(J), numerical(A)}.
% {body(I,J,decrease(A)) : rule(I), conjunction(J), numerical(A)}.

% (BSC)
% {body(I,J,enriched(P)) : rule(I), conjunction(J), pathway(P)}.
% {body(I,J,not_enriched(P)) : rule(I), conjunction(J), pathway(P)}.

{conjunction(1..3)}.

%--------------------------------------------
% Used to minimize used attributes.
%--------------------------------------------
used_attribute(A) :- body(_,_,at_least(A,_)).
used_attribute(A) :- body(_,_,at_most(A,_)).
% used_attribute(A) :- body(_,_,equals(A,_)).
% used_attribute(A) :- body(_,_,neg(A,_)).
used_attribute(A) :- body(_,_,lt(A,_)).
used_attribute(A) :- body(_,_,lt(_,A)).

% used_attribute(A) :- body(_,_,enriched(A)).
% used_attribute(A) :- body(_,_,not_enriched(A)).

#minimize{1@3,Seq: falseNegative(Seq)}.
#minimize{1@3,Seq: falsePositive(Seq)}.
#minimize{C@2,I,J,F: body(I,J,F), cost(F,C)}.
#minimize{1@2,X: used_attribute(X)}.

%----------------------------------------------
% Used for learning in a positive-only setting.
%----------------------------------------------
% :- #count{F: rule(F)} < 6.
% guard_length(I,J,N) :- rule(I), conjunction(J), #count{F: body(I,J,F)} = N.
% :- guard_length(I,J,N), N < 3.

% disjunctive_guard(I) :- rule(I), #count{F,J: body(I,J,F), J > 1} > 1.
% :- #count{I: disjunctive_guard(I)} = 0.

% This is useful for experimenting with MNIST.
% :- #count{S: state(S)} < 4.

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
