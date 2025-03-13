%*
%------- %
% CAVIAR %
%------- %
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
symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).
*%