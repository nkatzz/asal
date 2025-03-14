%------------------------------------------------------------------------------------------
% Prune solutions with useless transitions, i.e. transitions without a corresponding guard.
% Also, prune solutions that include guard body atoms for guards that do not appear in any
% transition fact in the solution.
%------------------------------------------------------------------------------------------
:- transition(S1,F,S2), S1 != S2, not rule(F).
:- body(I,_,_), not transition(_,I,_).
:- transition(I,f(I,J),J), I != J, not body(f(I,J),_,_).  % #count{J,F: body(I,J,F)} = 0

%---------------------------------------------------------------------------------------------
% Remove redundant stuff like
% f(1,2) :- a, not f(1,3).
% f(1,3) :- a.
% The first rule here is never satisfied, yet, it might be generated unless explicitly pruned.
%---------------------------------------------------------------------------------------------
:- body(f(S,S1),_,A), body(f(S,S2),_,A), S1 != S2.

%---------------------------------------------------------------------------------------------
% Avoid generating unsatisfiable rules like:
% holds(f(2,4),S,T) :- holds(equals(even,1),S,T), holds(equals(odd,1),S,T).
% Such rules may be generated during the generate phase, unless explicitly pruned.
%---------------------------------------------------------------------------------------------
:- body(I,J,F1), body(I,J,F2), F1 != F2, #count{S,T: holds(F1,S,T), holds(F2,S,T)} = 0.

%----------------------------------------------------------------------------------------------
% Always include self-loops for all states that appear in the automaton (skip-till-next-match).
%----------------------------------------------------------------------------------------------
:- state(S), not transition(S,_,S).

%---------------------------------------------------------------------------------------------
% Symmetry breaking. Impose an ordering on the used states, so that new states are introduced
% on demand only when previous states have already been already used. 
% This is to avoid symmetric solutions. Note that 1 is always the start state.
%---------------------------------------------------------------------------------------------
:- state(S), S != 1, S != 2, not accepting(S), #count{S1: state(S1), S1 = S - 1, S1 != 1} = 0.
:- transition(I,f(I,K),K), state(J), I < J, J < K, not transition(I,f(I,J),J).

% Just for debugging, to see if backward transitions (e.g. f(2,1)) are enabled.
% :- #count{I,J: transition(J,f(J,I),I), I < J} = 0.

%---------------------------------------------------------------------------------------------
% Symmetry breaking. Impose an ordering on the conjunction ids to avoid symmetric solutions.
%---------------------------------------------------------------------------------------------
:- body(I,J,_), J != 1, #count{J1: body(I,J1,_), J1 < J} = 0.

%---------------------------------------------------------------------------------------------
% Prune automata with paths that do not end to the accepting state.
%---------------------------------------------------------------------------------------------
reachable_from(X,Y) :- transition(X,_,Y).
reachable_from(X,Z) :- reachable_from(X,Y), reachable_from(Y,Z).
stranded_state(X) :- reachable_from(1,X), not reachable_from(X,S), accepting(S).
:- stranded_state(X), state(X).

%---------------------------------------------------------------------------------------------
% Prune useless states that are not reachable from the start state.
%---------------------------------------------------------------------------------------------
:- not reachable_from(1,S), state(S).

%---------------------------------------------------------------------------------------------
% Prune automata with unreachable states.
%---------------------------------------------------------------------------------------------
unreachable_state(S) :- state(S), not start(S), #count{S1: S1 != S, transition(S1,_,S)} = 0.
:- unreachable_state(S).

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

% Determinism constraint (just for sanity check).
:- inState(Seq,S,T), inState(Seq,S1,T), S != S1.
