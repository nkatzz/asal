%*
#script (python)
import string
import clingo
from clingo.symbol import Number

def get_symbol(symbol_index):
    x = int(str(symbol_index))
    alphabet = list(string.ascii_lowercase)
    return clingo.Function(alphabet[x - 1], [])

def get_index(symbol):
    alphabet = list(string.ascii_lowercase)
    return Number(int(alphabet.index(str(symbol))))

#end.

alphabetNum(1..10).
*%


num(0..9).


% attribute(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
% categorical(hus1b; slc22a1; loc100132354; slc22a16; abca12, c10orf41; c4orf6; c9orf129; cd1c; trim36; afap1l1; c6orf176; cabyr; ccdc146; ces8).
% numerical(none).

gene(X) :- seq(_,obs(X,_),_).
attribute(X) :- gene(X).
categorical :- attribute(X).
numerical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(V) :- num(V). 

%symbol(@get_symbol(Y)) :- alphabetNum(Y).
