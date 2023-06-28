%------------------------------------------------------------------------------------------
% Prune solutions with useless transitions, i.e. transitions without a corresponding guard.
% Also, prune solutions that include guard body atoms for guards that do not appear in any
% transition fact in the solution.
%------------------------------------------------------------------------------------------
:- transition(S1,F,S2), S1 != S2, not rule(F).
:- body(I,_,_), not transition(_,I,_).
:- transition(I,f(I,J),J), I != J, not body(f(I,J),_,_).  % #count{J,F: body(I,J,F)} = 0

%----------------------------------------------------------------------------------------------
% Always include self-loops for all states that appear in the automaton (skip-till-next-match).
%----------------------------------------------------------------------------------------------
:- state(S), not transition(S,_,S).

%---------------------------------------------------------------------------------------------
% Impose an ordering on the used states, so that new states are introduced on demand only
% when previous states have already been already used. This is to avoid symmetric solutions.
% Note that 1 is always the start state.
%---------------------------------------------------------------------------------------------
:- state(S), S != 1, S != 2, not accepting(S), #count{S1: state(S1), S1 = S - 1, S1 != 1} = 0.

%---------------------------------------------------------------------------------------------
% Prune automata with paths that do not end to the accepting state.
%---------------------------------------------------------------------------------------------
reachable_from(X,Y) :- transition(X,_,Y).
reachable_from(X,Z) :- reachable_from(X,Y), reachable_from(Y,Z).
stranded_state(X) :- reachable_from(1,X), not reachable_from(X,S), accepting(S).
:- stranded_state(X), state(X).

%---------------------------------------------------------------------------------------------
% Impose an ordering on the conjunction ids to avoid symmetric solutions.
%---------------------------------------------------------------------------------------------
:- body(I,J,_), J != 1, #count{J1: body(I,J1,_), J1 < J} = 0.

%---------------------------------------------------------------------------------------------
% Avoid solutions that contain redundancies, such as e.g.
% f(1,2) :- at_most(latitude,b).
% f(1,2) :- at_most(latitude,c).
%---------------------------------------------------------------------------------------------
:- body(F,_,at_most(A,X)), body(F,_,at_most(A,Y)), X != Y.
:- body(F,_,at_least(A,X)), body(F,_,at_least(A,Y)), X != Y.

%---------------------------------------------------------------------------------------------
% Constraints on things that shouldn't happen. The first two constraints simplify the search
% by excluding redundant solutions. The third constraint seems redundant, since initially
% it seems that the kind of contradicting conditions captured by the constraint cannot occur.
% They can, however, in order to invalidate, via specialization to an always false condition,
% guards that cover FPs. The fourth constraint is obvious.
%---------------------------------------------------------------------------------------------
:- body(I,J,at_most(A,V1)), body(I,J,at_most(A,V2)), V1 != V2.
:- body(I,J,at_least(A,V1)), body(I,J,at_least(A,V2)), V1 != V2.
:- body(I,J,at_least(A,V1)), body(I,J,at_most(A,V2)), V1 != V2.
:- body(I,J,equals(A,V1)), body(I,J,equals(A,V2)), V1 != V2.

unreachable_state(S) :- state(S), not start(S), #count{S1: S1 != S, transition(S1,_,S)} = 0.
:- unreachable_state(S).

% Determinism constraint (just for sanity check).
:- inState(Seq,S,T), inState(Seq,S1,T), S != S1.