
%----------------------------------%
% AVG robot (discretized PCA) %
%----------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(f1,X),T) :- seq(SeqId,f1(X),T).
seq(SeqId,obs(f2,X),T) :- seq(SeqId,f2(X),T).
seq(SeqId,obs(f3,X),T) :- seq(SeqId,f3(X),T).
seq(SeqId,obs(f4,X),T) :- seq(SeqId,f4(X),T).
seq(SeqId,obs(f5,X),T) :- seq(SeqId,f5(X),T).

attribute(f1; f2; f3; f4; f5).
numerical(f1; f2; f3; f4; f5).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
