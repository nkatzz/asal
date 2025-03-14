%*
%-------------------------------------------%
% AVG robot (coords and simple events only) %
%-------------------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(px,X),T) :- seq(SeqId,px(X),T).
seq(SeqId,obs(py,X),T) :- seq(SeqId,py(X),T).
seq(SeqId,obs(idle,X),T) :- seq(SeqId,idle(X),T).
seq(SeqId,obs(linear,X),T) :- seq(SeqId,linear(X),T).
seq(SeqId,obs(rotational,X),T) :- seq(SeqId,rotational(X),T).

attribute(px; py; pz; idle; linear; rotational).
numerical(px; py; pz).
categorical(idle; linear; rotational).

value(px,V) :- symbol(V).
value(py,V) :- symbol(V).
value(pz,V) :- symbol(V).
symbol(Y) :- alphabetNum(Y).

value(idle,1). value(idle,0). value(linear,1). value(linear,0). value(rotational,1). value(rotational,0).
*%