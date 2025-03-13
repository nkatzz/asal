
%------------------%
% BSC Kidney/Genes %
%------------------%
alphabetNum(0..9).
seq(SeqId,obs(hus1b,X),T) :- seq(SeqId,hus1b(X),T).
seq(SeqId,obs(slc22a1,X),T) :- seq(SeqId,slc22a1(X),T).
seq(SeqId,obs(loc100132354,X),T) :- seq(SeqId,loc100132354(X),T).
seq(SeqId,obs(slc22a16,X),T) :- seq(SeqId,slc22a16(X),T).
seq(SeqId,obs(abca12,X),T) :- seq(SeqId,abca12(X),T).
seq(SeqId,obs(c10orf41,X),T) :- seq(SeqId,c10orf41(X),T).
seq(SeqId,obs(c4orf6,X),T) :- seq(SeqId,c4orf6(X),T).
seq(SeqId,obs(c9orf129,X),T) :- seq(SeqId,c9orf129(X),T).
seq(SeqId,obs(cd1c,X),T) :- seq(SeqId,cd1c(X),T).
seq(SeqId,obs(trim36,X),T) :- seq(SeqId,trim36(X),T).
seq(SeqId,obs(afap1l1,X),T) :- seq(SeqId,afap1l1(X),T).
seq(SeqId,obs(c6orf176,X),T) :- seq(SeqId,c6orf176(X),T).
seq(SeqId,obs(cabyr,X),T) :- seq(SeqId,cabyr(X),T).
seq(SeqId,obs(ccdc146,X),T) :- seq(SeqId,ccdc146(X),T).
seq(SeqId,obs(ces8,X),T) :- seq(SeqId,ces8(X),T).


attribute(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
numerical(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
categorical(none).

% value(A,V) :- seq(S,obs(A,V),_), attribute(A).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
