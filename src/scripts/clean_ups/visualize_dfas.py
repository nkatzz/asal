import os
import time
import clingo
from visual_automata.fa.dfa import VisualDFA

s_time = time.time()
with open('/home/nkatz/Dropbox/ASP-experimentation/lookup.lp') as read:
    data = read.readlines()
    atoms = data[0].split(' ')
    for a in atoms:
        atom = clingo.parse_term(a.split('.')[0])
        # print(atom.arguments)
        b = atom.arguments[1]
        # print(b, b.type)
    print(time.time() - s_time)
read.close()
# rule = clingo.parse_term('p(a,b) :- q(a,b).')

dfa = VisualDFA(
    states={"q0", "q1", "q2", "q3", "q4"},
    input_symbols={"0", "1"},
    transitions={
        "q0": {"0": "q3", "1": "q1"},
        "q1": {"0": "q3", "1": "q2"},
        "q2": {"0": "q3", "1": "q2"},
        "q3": {"0": "q4", "1": "q1"},
        "q4": {"0": "q4", "1": "q1"},
    },
    initial_state="q0",
    final_states={"q2", "q4"},
)

"""def show_learnt(_dfa_: str):
    pass"""

print(os.getcwd())

dfa.show_diagram(view=True, filename='test', path=os.getcwd() + '/dfas', cleanup=True)
