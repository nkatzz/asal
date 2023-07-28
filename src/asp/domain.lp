#script (python)
import string
import clingo
from clingo.symbol import Number

def get_symbol(symbol_index):
    x = int(str(symbol_index))
    alphabet = list(string.ascii_lowercase)
    return clingo.Function(alphabet[x - 1], [])

def get_index(symbol):
    alphabet = list(string.ascii_lowercase)
    return Number(int(alphabet.index(str(symbol))))

#end.

% Maritime.
alphabetNum(1..8).
seq(SeqId,obs(longitude,X),T) :- seq(SeqId,longitude(X),T).
seq(SeqId,obs(latitude,X),T) :- seq(SeqId,latitude(X),T).
seq(SeqId,obs(speed,X),T) :- seq(SeqId,speed(X),T).
seq(SeqId,obs(heading,X),T) :- seq(SeqId,heading(X),T).
seq(SeqId,obs(course_over_ground,X),T) :- seq(SeqId,course_over_ground(X),T).
attribute(longitude; latitude; speed; heading; course_over_ground).
numerical(longitude; latitude; speed; heading; course_over_ground).
categorical(none).
% value(A,V) :- seq(S,obs(A,V),_), attribute(A).  % This is used in the generate part, it does not affect inference.
value(A,V) :- attribute(A), symbol(V).

%*
% CAVIAR
alphabetNum(1..10).
seq(SeqId,obs(p1,X),T) :- seq(SeqId,p1(X),T).
seq(SeqId,obs(p2,X),T) :- seq(SeqId,p2(X),T).
seq(SeqId,obs(eucl_dist,X),T) :- seq(SeqId,eucl_dist(X),T).
seq(SeqId,obs(orient_diff,X),T) :- seq(SeqId,orient_diff(X),T).
numerical(eucl_dist; orient_diff).
categorical(p1; p2).
value(eucl_dist,V) :- symbol(V).
value(orient_diff,V) :- symbol(V).
value(p1,walking). value(p1,active). value(p1,inactive).
value(p2,walking). value(p2,active). value(p2,inactive).
*%

%*
% BIO
alphabetNum(1..10).
seq(SeqId,obs(alive,X),T) :- seq(SeqId,alive(X),T).
seq(SeqId,obs(necrotic,X),T) :- seq(SeqId,necrotic(X),T).
seq(SeqId,obs(apoptotic,X),T) :- seq(SeqId,apoptotic(X),T).
numerical(alive; necrotic; apoptotic).
categorical(none).
value(A,V) :- attribute(A), symbol(V).
*%

symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).

attribute(A) :- numerical(A).
attribute(A) :- categorical(A).
:- numerical(A), categorical(A).

% holds(decrease(A),S,T) :- seq(S,obs(A,X),T), seq(S,obs(A,Y),T-1), X < Y, numerical(A).
% holds(increase(A),S,T) :- seq(S,obs(A,X),T), seq(S,obs(A,Y),T-1), X > Y, numerical(A).

holds(lt(A1,A2),SeqId,T) :- seq(SeqId,obs(A1,X),T), seq(SeqId,obs(A2,Y),T), X < Y, numerical(A1), numerical(A2), A1 != A2.
holds(at_most(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y <= X, symbol(X).   % , value(A,X)
holds(at_least(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y >= X, symbol(X).  % , value(A,X)
holds(equals(A,X),SeqId,T) :- seq(SeqId,obs(A,X),T), categorical(A).
% holds(neg(E,X),SeqId,T) :- seq(SeqId,obs(E,Y),T), value(E,X), Y != X.

% Costs for adding an atom to a transition rule.
cost(lt(A1,A2),1) :- attribute(A1), attribute(A2).
cost(at_most(A,V),1) :- value(A,V).
cost(at_least(A,V),1) :- value(A,V).
cost(equals(A,V),1) :- value(A,V).
cost(neg(A,V),2) :- value(A,V).

targetClass(1).
positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).

sequence(S) :- seq(S,_,_).
time(T) :- seq(_,_,T).
