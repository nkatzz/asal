%*
%----------%
% Weather  %
%----------%
alphabetNum(0..1).
seq(SeqId,obs(cloudy,X),T) :- seq(SeqId,cloudy(X),T).
seq(SeqId,obs(foggy,X),T) :- seq(SeqId,foggy(X),T).
seq(SeqId,obs(rainy,X),T) :- seq(SeqId,rainy(X),T).
seq(SeqId,obs(snowy,X),T) :- seq(SeqId,snowy(X),T).
seq(SeqId,obs(sunny,X),T) :- seq(SeqId,sunny(X),T).
attribute(cloudy; foggy; rainy; snowy; sunny).
numerical(none).
categorical(cloudy; foggy; rainy; snowy; sunny).
value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
*%