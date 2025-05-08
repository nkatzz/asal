from src.asal.template import Template
import clingo
from clingo.script import enable_python
import multiprocessing


def get_template(t: Template):
    return f"\n% SFA Induction template.\n{t.template}"


def get_domain(domain_file):
    with open(domain_file, "r", encoding="utf-8") as file:
        content = file.read()
    return f"\n% Domain specification:\n{content}"


def get_induction_program(args, t: Template, existing_model=None):
    if existing_model is None:
        existing_model = []
    program = []
    program.extend([get_interpreter(args), get_template(t),
                    state_defs, performance_defs, pos_neg_defs, performance_counts])
    program.append("\n% Max number of disjunctive alternatives per transition guard.")
    program.append(generate_conjs(args.max_alts))
    program.append('\n% Minimize false positive and false negative rates.')
    program.append(minimize_fps_fns(args))
    program.append('\n% Minimize size.')
    program.append(minimize_size)
    program.append(minimize_states)
    if args.min_attrs:
        program.append(mimimize_used_atts)
    program.append('\n')
    program.append(get_target_class(args.tclass))
    program.append(attribute_defs)
    if "equals" in args.predicates:
        program.extend(['\n' + def_equals, generate_equals, used_atts_equals, cost_equals])
    if "at_least" in args.predicates:
        program.extend(['\n' + def_at_least, generate_at_least, used_atts_at_least, cost_at_least])
    if "at_most" in args.predicates:
        program.extend(['\n' + def_at_most, generate_at_most, used_atts_at_most, cost_at_most])
    if "lt" in args.predicates:
        program.extend(['\n' + def_lt, generate_lt, used_atts_lt, cost_lt])
    if "neg" in args.predicates:
        program.extend(['\n' + def_neg, generate_neg, used_atts_neg, cost_neg])
    if "increase" in args.predicates:
        program.extend(['\n' + def_increase, generate_increase, used_atts_increase, cost_increase])
    if "decrease" in args.predicates:
        program.extend(['\n' + def_decrease, generate_decrease, used_atts_decrease, cost_decrease])
    program.append(get_domain(args.domain))

    if existing_model:  # is not None and isinstance(existing_model, list)
        existing = '\n'.join(existing_model)
        program.append(f"""\n% Existing model:\n{existing}""")

    program.append('\n\n% Redundancy constraints:\n' + base_constraints)
    if "at_most" in args.predicates or "at_least" in args.predicates:
        program.append(extra_constraints)
    program.append(induce_show)
    program = '\n'.join(program)
    return program


def get_test_program(args):
    program = []
    program.extend([get_interpreter(args), pos_neg_defs])
    program.append(get_domain(args.domain))
    program.append(get_target_class(args.tclass))

    if "equals" in args.predicates:
        program.extend(['\n' + def_equals])
    if "at_least" in args.predicates:
        program.extend(['\n' + def_at_least])
    if "at_most" in args.predicates:
        program.extend(['\n' + def_at_most])
    if "lt" in args.predicates:
        program.extend(['\n' + def_lt])
    if "neg" in args.predicates:
        program.extend(['\n' + def_neg])
    if "increase" in args.predicates:
        program.extend(['\n' + def_increase])
    if "decrease" in args.predicates:
        program.extend(['\n' + def_decrease])

    program.append(test_show)
    program = '\n'.join(program)
    return program


def get_interpreter(args):
    w = f'weight(S,{args.unsat_weight}) :- sequence(S).' if args.unsat_weight != 0 else ''
    interpreter = f"""\
    % SFA Interpreter.
    inState(SeqId,1,T) :- sequence(SeqId), seqStart(SeqId,T).
    inState(SeqId,S2,T+1) :- inState(SeqId,S1,T), transition(S1,F,S2), holds(F,SeqId,T).
    accepted(SeqId) :- inState(SeqId,S,T), accepting(S), seqEnd(SeqId,T).
    reach_accept_at(SeqId,T) :- inState(SeqId,S,T), accepting(S); #false: inState(SeqId,S,T1), T1 < T.
    
    % seqEnd/2 definition (used by the interpreter). 
    seq(SeqId,T) :- seq(SeqId,_,T).
    seqEnd(SeqId,T+1) :- seq(SeqId,T), not seq(SeqId,T+1).
    seqStart(SeqId,T) :- seq(SeqId,T), not seq(SeqId,T-1).
    
    sequence(S) :- seq(S,_,_).
    time(T) :- seq(_,_,T).
    {w}
    """
    return interpreter

state_defs = """\n
start(1).
state(S) :- transition(S,_,_).
state(S) :- transition(_,_,S).
state(S) :- rule(f(S,_)).
state(S) :- rule(f(_,S)).
"""

pos_neg_defs = """\
% Positive/negative sequence definitions.
positive(SeqId) :- class(SeqId,X), targetClass(X).
negative(SeqId) :- class(SeqId,X), not targetClass(X).
"""

performance_defs = """\
falseNegative(SeqId) :- positive(SeqId), not accepted(SeqId).
falsePositive(SeqId) :- negative(SeqId), accepted(SeqId).
truePositive(SeqId) :- positive(SeqId), accepted(SeqId).
"""

performance_counts = """\n
% Performance counts.
fns(X) :- X = #count{S: falseNegative(S)}.
fps(X) :- X = #count{S: falsePositive(S)}.
tps(X) :- X = #count{S: truePositive(S)}.
"""

induce_show = """
#show tps/1.
#show fps/1.
#show fns/1.
#show body/3.
#show transition/3.
#show accepting/1.
#show used_attribute/1.
#show rule/1.
"""

test_show = """
tp(SeqId) :- positive(SeqId), accepted(SeqId).
fp(SeqId) :- negative(SeqId), accepted(SeqId).
fn(SeqId) :- positive(SeqId), not accepted(SeqId).

#show tp/1.
#show fp/1.
#show fn/1.
"""

generate_at_most = "{body(I,J,at_most(A,V)) : rule(I), conjunction(J), numerical(A), value(A,V)}."
generate_at_least = "{body(I,J,at_least(A,V)) : rule(I), conjunction(J), numerical(A), value(A,V)}."
generate_lt = "{body(I,J,lt(A1,A2)) : rule(I), conjunction(J), numerical(A1), numerical(A2), A1 != A2}."
generate_equals = "{body(I,J,equals(A,V)) : rule(I), conjunction(J), categorical(A), value(A,V)}."
generate_neg = "{body(I,J,neg(A,V)) : rule(I), conjunction(J), categorical(A), value(A,V)}."
generate_increase = "{body(I,J,increase(A)) : rule(I), conjunction(J), numerical(A)}."
generate_decrease = "{body(I,J,decrease(A)) : rule(I), conjunction(J), numerical(A)}."


def generate_conjs(max_alts):
    return f"{{conjunction(1..{max_alts})}}."


used_atts_at_least = "used_attribute(A) :- body(_,_,at_least(A,_))."
used_atts_at_most = "used_attribute(A) :- body(_,_,at_most(A,_))."
used_atts_equals = "used_attribute(A) :- body(_,_,equals(A,_))."
used_atts_neg = "used_attribute(A) :- body(_,_,neg(A,_))."
used_atts_lt = """\
used_attribute(A) :- body(_,_,lt(A,_)).
used_attribute(A) :- body(_,_,lt(_,A)).
"""
used_atts_increase = "used_attribute(A) :- body(_,_,increase(A))."
used_atts_decrease = "used_attribute(A) :- body(_,_,decrease(A))."

"""
def minimize_fps_fns(coverage_first=False, unsat_weight=1):
    c = []
    if coverage_first:
        c.append("#minimize{1@1,Seq: falseNegative(Seq)}.")
        c.append("#minimize{1@1,Seq: falsePositive(Seq)}.")
    else:
        c.append("#minimize{1@0,Seq: falseNegative(Seq)}.")
        c.append("#minimize{1@0,Seq: falsePositive(Seq)}.")
    return '\n'.join(c)
"""

#"""
def minimize_fps_fns(args):
    c = ["satisfied(Seq) :- positive(Seq), accepted(Seq).", "satisfied(Seq) :- negative(Seq), not accepted(Seq)."]
    weight = args.unsat_weight if args.unsat_weight != 0 else 'W'
    level = 1 if args.coverage_first else 0
    c.append(f"#minimize{{{weight}@{level},Seq: sequence(Seq), weight(Seq,W), not satisfied(Seq)}}.")
    return '\n'.join(c)
#"""

minimize_size = "#minimize{C@0,I,J,F: body(I,J,F), cost(F,C)}."
mimimize_used_atts = "#minimize{1@0,X: used_attribute(X)}."
minimize_states = "#minimize{1@0,X: state(X)}."


def get_target_class(tc):
    return f"\ntargetClass({tc})."


attribute_defs = """\n
attribute(A) :- numerical(A).
attribute(A) :- categorical(A).
:- numerical(A), categorical(A).
"""

def_at_most = "holds(at_most(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y <= X, symbol(X)."
def_at_least = "holds(at_least(A,X),SeqId,T) :- seq(SeqId,obs(A,Y),T), numerical(A), Y >= X, symbol(X)."
def_lt = ("holds(lt(A1,A2),SeqId,T) :- seq(SeqId,obs(A1,X),T), seq(SeqId,obs(A2,Y),T), X < Y, "
          "numerical(A1), numerical(A2), A1 != A2.")
def_equals = "holds(equals(A,X),SeqId,T) :- seq(SeqId,obs(A,X),T), categorical(A), A != none."
def_neg = "holds(neg(E,X),SeqId,T) :- seq(SeqId,obs(E,Y),T), value(E,X), Y != X."
def_decrease = "holds(decrease(A),S,T) :- seq(S,obs(A,X),T), seq(S,obs(A,Y),T-1), X < Y, numerical(A)."
def_increase = "holds(increase(A),S,T) :- seq(S,obs(A,X),T), seq(S,obs(A,Y),T-1), X > Y, numerical(A)."

# Costs can be modified here.
cost_lt = "cost(lt(A1,A2),1) :- attribute(A1), attribute(A2)."
cost_at_most = "cost(at_most(A,V),1) :- value(A,V)."
cost_at_least = "cost(at_least(A,V),1) :- value(A,V)."
cost_equals = "cost(equals(A,V),1) :- value(A,V)."
cost_neg = "cost(neg(A,V),1) :- value(A,V)."
cost_increase = "cost(increase(A),1) :- numerical(A)."
cost_decrease = "cost(decrease(A),1) :- numerical(A)."

base_constraints = """\
%------------------------------------------------------------------------------------------
% Prune solutions with useless transitions, i.e. transitions without a corresponding guard.
% Also, prune solutions that include guard body atoms for guards that do not appear in any
% transition fact in the solution.
%------------------------------------------------------------------------------------------
:- transition(S1,F,S2), S1 != S2, not rule(F).
:- body(I,_,_), not transition(_,I,_).
:- transition(I,f(I,J),J), I != J, not body(f(I,J),_,_).  % #count{J,F: body(I,J,F)} = 0

%---------------------------------------------------------------------------------------------
% Remove redundant stuff like
% f(1,2) :- a, not f(1,3).
% f(1,3) :- a.
% The first rule here is never satisfied, yet, it might be generated unless explicitly pruned.
%---------------------------------------------------------------------------------------------
:- body(f(S,S1),_,A), body(f(S,S2),_,A), S1 != S2.

%---------------------------------------------------------------------------------------------
% Avoid generating unsatisfiable rules like:
% holds(f(2,4),S,T) :- holds(equals(even,1),S,T), holds(equals(odd,1),S,T).
% Such rules may be generated during the generate phase, unless explicitly pruned.
%---------------------------------------------------------------------------------------------
% Update (8/5/2025): This is a very expensive constraint, blows up the grounding time. 
% :- body(I,J,F1), body(I,J,F2), F1 != F2, #count{S,T: holds(F1,S,T), holds(F2,S,T)} = 0.

%----------------------------------------------------------------------------------------------
% Always include self-loops for all states that appear in the automaton (skip-till-next-match).
%----------------------------------------------------------------------------------------------
:- state(S), not transition(S,_,S).

%---------------------------------------------------------------------------------------------
% Symmetry breaking. Impose an ordering on the used states, so that new states are introduced
% on demand only when previous states have already been already used. 
% This is to avoid symmetric solutions. Note that 1 is always the start state.
%---------------------------------------------------------------------------------------------
:- state(S), S != 1, S != 2, not accepting(S), #count{S1: state(S1), S1 = S - 1, S1 != 1} = 0.
:- transition(I,f(I,K),K), state(J), I < J, J < K, not transition(I,f(I,J),J).

% Just for debugging, to see if backward transitions (e.g. f(2,1)) are enabled.
% :- #count{I,J: transition(J,f(J,I),I), I < J} = 0.

%---------------------------------------------------------------------------------------------
% Symmetry breaking. Impose an ordering on the conjunction ids to avoid symmetric solutions.
%---------------------------------------------------------------------------------------------
:- body(I,J,_), J != 1, #count{J1: body(I,J1,_), J1 < J} = 0.

%---------------------------------------------------------------------------------------------
% Prune automata with paths that do not end to the accepting state.
%---------------------------------------------------------------------------------------------
reachable_from(X,Y) :- transition(X,_,Y).
reachable_from(X,Z) :- reachable_from(X,Y), reachable_from(Y,Z).
stranded_state(X) :- reachable_from(1,X), not reachable_from(X,S), accepting(S).
:- stranded_state(X), state(X).

%---------------------------------------------------------------------------------------------
% Prune useless states that are not reachable from the start state.
%---------------------------------------------------------------------------------------------
:- not reachable_from(1,S), state(S).

%---------------------------------------------------------------------------------------------
% Prune automata with unreachable states.
%---------------------------------------------------------------------------------------------
unreachable_state(S) :- state(S), not start(S), #count{S1: S1 != S, transition(S1,_,S)} = 0.
:- unreachable_state(S).

% Determinism constraint (just for sanity check).
:- inState(Seq,S,T), inState(Seq,S1,T), S != S1.

:- body(I,J,equals(A,V1)), body(I,J,equals(A,V2)), V1 != V2.
"""

extra_constraints = """\
%---------------------------------------------------------------------------------------------
% Avoid solutions that contain redundancies, such as e.g.
% f(1,2) :- at_most(latitude,b).
% f(1,2) :- at_most(latitude,c).
%---------------------------------------------------------------------------------------------
:- body(F,I,at_most(A,X)), body(F,J,at_most(A,Y)), X != Y.
:- body(F,I,at_least(A,X)), body(F,J,at_least(A,Y)), X != Y.

%---------------------------------------------------------------------------------------------
% Constraints on things that shouldn't happen. The first two constraints simplify the search
% by excluding redundant solutions. The third constraint seems redundant, since initially
% it seems that the kind of contradicting conditions captured by the constraint cannot occur.
% They can, however, in order to invalidate, via specialization to an always false condition,
% guards that cover FPs.
%---------------------------------------------------------------------------------------------
:- body(I,J,at_most(A,V1)), body(I,J,at_most(A,V2)), V1 != V2.
:- body(I,J,at_least(A,V1)), body(I,J,at_least(A,V2)), V1 != V2.
:- body(I,J,at_least(A,V1)), body(I,J,at_most(A,V2)), V1 != V2.
"""
