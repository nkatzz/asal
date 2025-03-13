
%----------------------------------%
% AVG robot (latent features) %
%----------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(latent_0,X),T) :- seq(SeqId,latent_0(X),T).
seq(SeqId,obs(latent_1,X),T) :- seq(SeqId,latent_1(X),T).
seq(SeqId,obs(latent_2,X),T) :- seq(SeqId,latent_2(X),T).
seq(SeqId,obs(latent_3,X),T) :- seq(SeqId,latent_3(X),T).

attribute(latent_0; latent_1; latent_2; latent_3).
numerical(latent_0; latent_1; latent_2; latent_3).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
