num(0..10).

% attribute(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
% numerical(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
% categorical(none).

attribute(pc1; pc2; pc3; pc4; pc5; pc6; pc7; pc8; pc9; pc10).
numerical(pc1; pc2; pc3; pc4; pc5; pc6; pc7; pc8; pc9; pc10).
categorical(none).

% numerical(X) :- seq(_,obs(X,_),_).
% categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(S) :- num(S). 

