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

alphabetNum(1..24).

attribute(dist).
numerical(dist).

% attribute(px_1; py_1; yaw_1; px_2; py_2; yaw_2; vx_1; vy_1; vx_2; vy_2).
% categorical(px_1; py_1; yaw_1; px_2; py_2; yaw_2; vx_1; vy_1; vx_2; vy_2).

% attribute(px_1; py_1; px_2; py_2; yaw_1; yaw_2; dist).
% numerical(px_1; py_1; px_2; py_2; yaw_1; yaw_2; dist).

categorical(none).

value(A,V) :- attribute(A), symbol(V).
symbol(@get_symbol(Y)) :- alphabetNum(Y).
% index(Y,@get_index(Y)) :- symbol(Y).


