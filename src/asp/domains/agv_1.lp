
%----------------------------------%
% AVG robot (all spatial features) %
%----------------------------------%
alphabetNum(0..9).
seq(SeqId,obs(px,X),T) :- seq(SeqId,px(X),T).
seq(SeqId,obs(py,X),T) :- seq(SeqId,py(X),T).
seq(SeqId,obs(pz,X),T) :- seq(SeqId,pz(X),T).
seq(SeqId,obs(ox,X),T) :- seq(SeqId,ox(X),T).
seq(SeqId,obs(oy,X),T) :- seq(SeqId,oy(X),T).
seq(SeqId,obs(oz,X),T) :- seq(SeqId,oz(X),T).
seq(SeqId,obs(ow,X),T) :- seq(SeqId,ow(X),T).
seq(SeqId,obs(vx,X),T) :- seq(SeqId,vx(X),T).
seq(SeqId,obs(vy,X),T) :- seq(SeqId,vy(X),T).
seq(SeqId,obs(vz,X),T) :- seq(SeqId,vz(X),T).

attribute(px; py; pz; ox; oy, oz; ow; vx; vy; vz).
numerical(px; py; pz; ox; oy, oz; ow; vx; vy; vz).

% attribute(px; py; ox; oy; vx; vy).
% numerical(px; py; ox; oy; vx; vy).

% attribute(px; py; ox; vx; vy).
% numerical(px; py; ox; vx; vy).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(Y) :- alphabetNum(Y).
