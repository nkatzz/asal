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


%-----%
% BIO %
%-----%
alphabetNum(1..10).
seq(SeqId,obs(alive,X),T) :- seq(SeqId,alive(X),T).
seq(SeqId,obs(necrotic,X),T) :- seq(SeqId,necrotic(X),T).
seq(SeqId,obs(apoptotic,X),T) :- seq(SeqId,apoptotic(X),T).
numerical(alive; necrotic; apoptotic).
categorical(none).
value(A,V) :- attribute(A), symbol(V).
symbol(@get_symbol(Y)) :- alphabetNum(Y).
index(Y,@get_index(Y)) :- symbol(Y).
