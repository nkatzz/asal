num(0..19).

attribute(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
categorical(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
numerical(none).

% categorical(X) :- seq(_,obs(X,_),_).
% numerical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(S) :- num(S). 

