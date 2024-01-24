type(X) :- agent_type(_, agent1(X)).
type(X) :- agent_type(_, agent2(X)).

action(X) :- seq(_,a1(action(X)),_).
action(X) :- seq(_,a2(action(X)),_).

location(X) :- seq(_,a1(location(X)),_).
location(X) :- seq(_,a2(location(X)),_).

pos(X) :- class(X, 1).
neg(X) :- class(X, 0).

#show pos/1.
% #show neg/1.

% #show action/1.
% #show location/1.
