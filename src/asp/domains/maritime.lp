
%----------%
% Maritime %
%----------%
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
symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).
