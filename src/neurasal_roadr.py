import sys
import os

# Need to add the project root in the pythonpath.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.asal_nesy.cirquits.build_sdds import SDDBuilder
from src.asal_nesy.cirquits.circuit_auxils import model_count_nnf, nnf_map_to_sfa
from src.asal_nesy.neurasal.utils import *

asp_program = \
    """
% The allowed values need to be alligned with what the NN predicts
% for each attribute (a1, a2, l1, l2).
 
1 {value(a1,stop) ; value(a1,movaway) ; value(a1,movtow) ; value(a1,other)} 1.
1 {value(a2,stop) ; value(a2,movaway) ; value(a2,movtow) ; value(a2,other)} 1.
1 {value(l1,incomlane) ; value(l1,jun) ; value(l1,vehlane) ; value(l1,other)} 1.
1 {value(l2,incomlane) ; value(l2,jun) ; value(l2,vehlane) ; value(l2,other)} 1.

equals(same_lane,true) :- value(l1,X), value(l2,X), X != other.
equals(same_lane,false) :- value(l1,X), value(l2,Y), X != Y.

equals(action_1,A) :- value(a1,A).
equals(action_2,A) :- value(a2,A).

equals(location_1,L) :- value(l1,L).
equals(location_2,L) :- value(l2,L).
"""

automaton = \
    """
f(1,4) :- equals(same_lane,true), equals(action_2,movaway).
f(1,3) :- equals(same_lane,false), not f(1,4).
f(2,4) :- equals(action_1,stop), equals(location_2,incomlane).
f(3,1) :- equals(action_1,movaway).
f(1,2) :- equals(action_2,movtow), not f(1,3), not f(1,4).
f(2,3) :- equals(action_2,movtow), not f(2,4).
f(4,4) :- #true.
f(1,1) :- not f(1,2), not f(1,3), not f(1,4).
f(3,3) :- not f(3,1).
f(2,2) :- not f(2,3), not f(2,4).
"""

# Necessary for compiling the automaton's rules into circuits
query_defs = \
    """
    query(guard,f(1,2)) :- f(1,2).
    query(guard,f(1,3)) :- f(1,3).
    query(guard,f(3,1)) :- f(3,1).
    query(guard,f(2,3)) :- f(2,3).
    query(guard,f(1,4)) :- f(1,4).
    query(guard,f(2,4)) :- f(2,4).
    query(guard,f(1,1)) :- f(1,1).
    query(guard,f(2,2)) :- f(2,2).
    query(guard,f(3,3)) :- f(3,3).
    query(guard,f(4,4)) :- f(4,4).

    #show value/2.
    #show query/2.
    """


def compile_roadr_sfa():
    asp_compilation_program = asp_program + automaton + query_defs
    sdd_builder = SDDBuilder(asp_compilation_program,
                             vars_names=['a1', 'a2', 'l1', 'l2'],
                             categorical_vars=['a1', 'a2', 'l1', 'l2'],
                             clear_fields=False)
    sdd_builder.build_nnfs()
    circuits = sdd_builder.circuits
    sfa = nnf_map_to_sfa(circuits)
    return sfa


if __name__ == "__main__":
    sfa = compile_roadr_sfa()
    print(f'\nCompiled SFA: {sfa.transitions}\nStates: {sfa.states}\nSymbols: {sfa.symbols}')
