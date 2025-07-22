import nnf
from src.asal_nesy.deepfa.automaton import DeepFA
from src.asal_nesy.neurasal.mnist.compile_multivar_asp import compile_sfa

def generate_constraint(values):
    import itertools

    return nnf.And((~i | ~j) for i, j in itertools.combinations(values, r=2)) & nnf.Or(
        values
    )


variables = {}
for i in range(1, 2):  # single-var SFA
    variables[i] = [nnf.Var("d{}_{}".format(i, j)) for j in range(10)]  # 10 is the number of classes.

constraint = nnf.And(map(generate_constraint, variables.values()))

def even(digit: int):
    return nnf.Or([var for var in variables[digit] if int(var.name[-1]) % 2 == 0])

def odd(digit: int):
    return nnf.Or([var for var in variables[digit] if int(var.name[-1]) % 2 != 0])

def leq3(digit: int):
    return nnf.Or([var for var in variables[digit] if int(var.name[-1]) <= 3])

def gt3(digit: int):
    return nnf.Or([var for var in variables[digit] if int(var.name[-1]) > 3])

def leq6(digit: int):
    return nnf.Or([var for var in variables[digit] if int(var.name[-1]) <= 6])

def gt6(digit: int):
    return nnf.Or([var for var in variables[digit] if int(var.name[-1]) > 6])

d = {'even(1)': even(1), 'odd(1)': odd(1), 'gt6(1)': gt6(1), 'leq6(1)': leq6(1), 'gt3(1)': gt3(1),
     'leq3(1)': leq3(1)}

"""
accepting(4).
transition(1,f(1,1),1). transition(1,f(1,2),2). transition(2,f(2,2),2). 
transition(2,f(2,3),3). transition(3,f(3,3),3). transition(3,f(3,4),4). 
transition(4,f(4,4),4).

holds(f(1,2),T) :- holds(equals(d1,even),T), holds(equals(d1,gt_6),T).
holds(f(2,3),T) :- holds(equals(d1,odd),T), holds(equals(d1,leq_6),T).
holds(f(3,4),T) :- holds(equals(d1,leq_3),T).
holds(f(1,1),T) :- time(T), not holds(f(1,2),T).
holds(f(2,2),T) :- time(T), not holds(f(2,3),T).
holds(f(3,3),T) :- time(T), not holds(f(3,4),T).

holds(f(4,4),T) :- time(T).  % No over-general self loop to avoid having acceptance prob. increase indefinitely in the experiments.
% holds(f(4,4),T) :- holds(equals(d1,leq_3),T).
% holds(f(4,1),T) :- time(T), not holds(f(4,4),T).
"""

f12 = d['even(1)'] & d['gt6(1)']
f23 = d['odd(1)'] & d['leq6(1)']
f34 = d['leq3(1)']
# f44 = d['leq3(1)']
# f41 = f44.negate()
f11 = f12.negate()
f22 = f23.negate()
f33 = f34.negate()

"""
def get_sfa():
    deepfa = DeepFA(
        {
            1: {1: f11 & constraint, 2: f12 & constraint},
            2: {2: f22 & constraint, 3: f23 & constraint},
            3: {3: f33 & constraint, 4: f34 & constraint,},
            4: {4: f44 & constraint, 1: f41 & constraint},
        },
        1,
        {4},
    )
    return deepfa
"""

def get_sfa():
    deepfa = DeepFA(
        {
            1: {1: f11 & constraint, 2: f12 & constraint},
            2: {2: f22 & constraint, 3: f23 & constraint},
            3: {3: f33 & constraint, 4: f34 & constraint,},
            4: {4: nnf.true & constraint},
        },
        1,
        {4},
    )
    return deepfa

if __name__ == '__main__':
    deepfa = get_sfa()
    print(f'\nCompiled SFA: {deepfa.transitions}\nStates: {deepfa.states}\nSymbols: {deepfa.symbols}')