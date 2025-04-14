from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import model_count_nnf, nnf_map_to_sfa

asp_program = \
    """
    1 {value(d1,0..9)} 1.
    1 {value(d2,0..9)} 1.
    1 {value(d3,0..9)} 1.
    even(D) :- value(D,N), N \ 2 = 0.
    odd(D) :- value(D,N), N \ 2 != 0.
    larger_than_6(D) :- value(D,N), N > 6.
    less_eq_6(D) :- value(D,N), N <= 6.
    larger_than_3(D) :- value(D,N), N > 3.
    less_eq_3(D) :- value(D,N), N <= 3.
    larger_than_2(D) :- value(D,N), N > 2.
    less_eq_2(D) :- value(D,N), N <= 2.
    larger_than_4(D) :- value(D,N), N > 4.
    less_eq_4(D) :- value(D,N), N <= 4.
    larger_than_5(D) :- value(D,N), N > 5.
    less_eq_5(D) :- value(D,N), N <= 5.
    larger_than_7(D) :- value(D,N), N > 7.
    less_eq_7(D) :- value(D,N), N <= 7.

    equals(even(D),1) :- even(D).
    equals(odd(D),1) :- odd(D).
    equals(gt_6(D),1) :- larger_than_6(D).
    equals(leq_6(D),1) :- less_eq_6(D).
    equals(gt_2(D),1) :- larger_than_2(D).
    equals(leq_2(D),1) :- less_eq_2(D).
    equals(gt_3(D),1) :- larger_than_3(D).
    equals(leq_3(D),1) :- less_eq_3(D).
    equals(gt_4(D),1) :- larger_than_4(D).
    equals(leq_4(D),1) :- less_eq_4(D).
    equals(gt_5(D),1) :- larger_than_5(D).
    equals(leq_5(D),1) :- less_eq_5(D).
    equals(gt_7(D),1) :- larger_than_7(D).
    equals(leq_7(D),1) :- less_eq_7(D).
    """

automaton = \
"""
f(2,3) :- equals(even(d1),1), equals(gt_6(d2),1).
f(2,1) :- equals(gt_6(d3),1), not f(2,3).
f(1,2) :- equals(leq_6(d1),1).
f(3,1) :- equals(gt_5(d3),1), not f(3,4).
f(3,4) :- equals(leq_5(d1),1), equals(leq_5(d3),1).
f(2,2) :- not f(2,1), not f(2,3).
f(3,3) :- not f(3,1), not f(3,4).
f(1,1) :- not f(1,2).
f(4,4) :- #true.
"""

query_defs = \
    """
    query(guard,f(1,2)) :- f(1,2).
    query(guard,f(2,1)) :- f(2,1).
    query(guard,f(3,1)) :- f(3,1).
    query(guard,f(2,3)) :- f(2,3).
    query(guard,f(3,4)) :- f(3,4).
    query(guard,f(1,1)) :- f(1,1).
    query(guard,f(2,2)) :- f(2,2).
    query(guard,f(3,3)) :- f(3,3).
    query(guard,f(4,4)) :- f(4,4).

    #show value/2.
    #show query/2.
    """

def compile_sfa():
    import sys
    sys.setrecursionlimit(500000)
    asp_compilation_program = asp_program + automaton + query_defs
    sdd_builder = SDDBuilder(asp_compilation_program,
                             vars_names=['d1', 'd2', 'd3'],
                             categorical_vars=['d1', 'd2', 'd3'],
                             clear_fields=False)
    sdd_builder.build_nnfs()
    circuits = sdd_builder.circuits
    sfa = nnf_map_to_sfa(circuits)
    return sfa

if __name__ == "__main__":
    sfa = compile_sfa()
    print(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')