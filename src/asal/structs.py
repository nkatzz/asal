from src.asal.auxils import precision, recall, f1, flatten
from src.asal.template import Template
from clingo.symbol import parse_term
from src.asal.logger import *
import random


class Automaton:

    def __init__(self, *args, deterministic=True):
        # print(args)

        """Set of transition facts as clingo objects."""
        self.transitions: list[TransitionAtom] = []

        """Set of transition facts as strings."""
        self.transitions_str = []
        self.accepting_states = []

        """The list of transition rules."""
        self.rules: list[Rule] = []

        """key: (h, c_id), value: [body_atoms], where h is the rule's head id (e.g. f(1,5)) and c_id
         is a conjunction id (so this represents a rule variant, since there may be multiple versions
         - variants - of the same rule). This is dictionary is used for the generation of the automaton
          for the first time, in order to progressively add body literals (as returned from the solver)
         to the bodies of rules, which are themselves gradually formed.
         This dictionary is of no used after the creation of the automaton instance."""
        self.__rules_dict = {}

        """key: a rule's head id (e.g. f(1,5)), body: rule ids that are to be added negated to the body
        of the rule represented by the key."""
        self.rules_mutex_conditions = {}

        """Similar to the above, but for self-loop guards."""
        self.self_loop_guards = {}

        """tp, fp and fn atoms from the mini-batch that the automaton
         has been generated from. For display only."""
        self.local_performance = []

        """Contains the body/3 atoms in a model. This is used to pass an existing model to the solver
        in a format that allows to properly reason with its revision (with the current encoding, 
        this is infeasible to do with the automaton in transitions/rules format)."""
        self.body_atoms_asp = []

        """Sequence (example) ids that correspond to TP, FP, FN predictions for this automaton"""
        self.tps_seq_ids = []
        self.fps_seq_ids = []
        self.fns_seq_ids = []

        """The cost returned by the solver."""
        self.cost = None

        """The corresponding variable returned by the solver for this model."""
        self.optimality_proven = False

        """F1-score on the training set"""
        self.global_performance = 0.0

        """(TP, FP, FN) counts on the training set"""
        self.global_performance_counts = (0, 0, 0)

        """TP & FP counts for each accepting path in the automaton"""
        self.scored_paths = []

        """(TP, FP, FN) counts per batch"""
        self.counts_per_batch = {}

        self.is_deterministic = deterministic
        self.is_empty = False
        self.to_string = ''

        if args.__len__() != 0:
            self.asp_model = args[0]
            self.body_atoms_asp = list(filter(lambda atom: isinstance(atom, GuardBodyAtom), self.asp_model))
            self.template: Template = args[1]
            self.__generate()
            self.__generate_final_rule_list()
            self.show(mode='reasoning')  # Call this once, since this call populates the guard dicts.
            # self.to_string = self.show(mode='reasoning')
        else:
            self.is_empty = True

    def __generate(self):
        """Generate an automaton from a Clingo model."""
        for atom in self.asp_model:
            if isinstance(atom, TransitionAtom):
                self.transitions.append(atom)
                self.transitions_str.append(atom.str)
            elif isinstance(atom, AcceptingStateAtom):
                self.accepting_states.append(atom.str + '.')
            elif isinstance(atom, GuardBodyAtom):
                rule_head = atom.guard_head
                body_id = atom.body_id
                body_atom = atom.body_atom
                if (rule_head, body_id) in self.__rules_dict:
                    self.__rules_dict[(rule_head, body_id)].append(body_atom)
                else:
                    self.__rules_dict[(rule_head, body_id)] = [body_atom]
            elif isinstance(atom, CountsAtom):
                self.local_performance.append(atom.str)
            else:
                pass

    def update_seqs_store(self, seqs):
        tp_seqs, fps_seqs, fns_seqs = seqs[0], seqs[1], seqs[2]
        self.tps_seq_ids = self.tps_seq_ids + tp_seqs
        self.fps_seq_ids = self.fps_seq_ids + fps_seqs
        self.fns_seq_ids = self.fns_seq_ids + fns_seqs

    def __get_used_transition_functions(self) -> set[str]:
        """Get the f(ij)'s that appear in the automaton's transition guards."""
        return set([atom.guard for atom in self.transitions])

    def __get_used_self_loop_functs(self) -> set[str]:
        """Get the f(ij)'s that appear in the automaton's transition guards, such that i == j."""
        return set([atom.guard for atom in self.transitions if self.__is_self_loop_id(atom.guard)])

    def __get_self_loop_keys(self):
        """Get all f(ij)'s in the template such that i == j."""
        return filter(lambda x: self.__is_self_loop_id(x), self.template.mutex_map.keys())

    def __get_mutex_conditions(self, rule_head: str) -> list[str]:
        """Get the 'not f(ij)' mutex conditions for the funct guard."""
        return self.template.mutex_map[rule_head]

    def __get_guards(self, mode='reasoning'):
        """
        Form transition guard rules by joining the list of literals that correspond to the body of a rule and a list
        of literals that are supposed to be negated in the rule body, to ensure mutual exclusivity of the
        transition guards.
        """
        rules = []
        for rule in self.rules:

            f, body_atoms = rule.head, rule.body

            if self.is_deterministic:
                mutex_conditions = [x for x in self.__get_mutex_conditions(f) if
                                    x in self.__get_used_transition_functions()]

                join_body = ', '.join(body_atoms) if mode == 'simple' else \
                    ', '.join(map(lambda x: f'holds({x},S,T)', body_atoms))

                join_mutexes = ', '.join(map(lambda x: f'not {x}', mutex_conditions)) if mode == 'simple' else \
                    ', '.join(map(lambda x: f'not holds({x},S,T)', mutex_conditions))

                body = f'{join_body}, {join_mutexes}.' if mutex_conditions != [] else f'{join_body}.'
                rule = f'{f} :- {body}' if mode == 'simple' else f'holds({f},S,T) :- {body}'
                self.rules_mutex_conditions[f] = mutex_conditions
                rules.append(rule)
        return rules

    @staticmethod
    def __is_self_loop_id(f):
        """Check if for an f(ij) it holds that i == j."""

        def get_id(x): return x.split("(")[1].split(")")[0].split(',')

        def is_self_loop_id(x): return 1 if x[0] == x[1] else 0

        return is_self_loop_id(get_id(f))

    def __get_self_loop_guards(self, mode='reasoning'):
        """Get a string representation of self-loop transition guard rules."""
        all_self_loop_functs = set(self.__get_self_loop_keys())
        used_self_loop_functs = self.__get_used_self_loop_functs()
        self_loop_functs = all_self_loop_functs.intersection(used_self_loop_functs)
        self_loop_rules = []
        for f in self_loop_functs:
            mutex_conditions = [x for x in self.__get_mutex_conditions(f) if
                                x in self.__get_used_transition_functions()]
            self.self_loop_guards[f] = mutex_conditions
            if mode == 'reasoning':
                _head_ = 'holds({0},S,T)'.format(f)
                _body = ['sequence(S)', 'time(T)'] + \
                        list(map(lambda x: 'not holds({0},S,T)'.format(x), mutex_conditions))
            else:
                _head_ = f
                _body = list(map(lambda x: 'not {0}'.format(x), mutex_conditions))
            _body_ = _body[0] if len(_body) == 1 else ', '.join(_body)
            rule = '{0} :- {1}.'.format(_head_, _body_) if len(_body) > 0 else '{0} :- #true.'.format(_head_)
            self_loop_rules.append(rule)
        return self_loop_rules

    def show(self, mode='reasoning'):
        """Generate the transition guard rules for display and for using them in reasoning tasks."""
        guards = self.__get_guards(mode) + self.__get_self_loop_guards(mode)
        accepting = ' '.join(self.accepting_states)
        out = accepting + '\n' + ' '.join(t + '.' for t in map(lambda x: x.str, self.transitions)) + '\n' + '\n'.join(
            guards)
        self.to_string = out
        return out

    def __generate_final_rule_list(self):
        self.rules = [Rule(i, j[0], v) for
                      i, (j, v) in zip(range(1, len(self.__rules_dict) + 1), self.__rules_dict.items())]

    def __get_states(self):
        return sorted(list(set(flatten([[a.from_state, a.to_state] for a in self.transitions]))))

    def __get_outgoing_guards(self, state):
        """Returns all outgoing guard ids for some given state."""
        get_from_state = lambda guard_id: guard_id.split(',')[0].split('(')[1].strip()
        return list(set([a.guard for a in self.transitions if get_from_state(a.guard) == state]))

    def __get_mutexes(self, guard):
        if guard in self.rules_mutex_conditions.items():
            return self.rules_mutex_conditions[guard]
        elif guard in self.self_loop_guards.items():
            return self.self_loop_guards[guard]
        else:
            logger.error(f'Guard {guard} cannot be found in mutexes dicts. The SFA is {self.show()}')

    def flatten_guards(self):
        """Gets the conditions that need to be (dis-)satisfied for a guard to be take effect,
           by "unfolding" the actual definition and mutual exclusivity conditions for each guard.
           This is used in the procedural evaluation of an SFA."""
        states = self.__get_states()
        for s in states:
            # Get all s-outgoing guards (these are the ones that have mutex dependencies)
            s_guards = self.__get_outgoing_guards(s)

            # start working through the list of guards dealing first with those that have no mutex
            # conditions. Those should be "resolved" first and their body conditions be used in
            # resolving other, more complex (w.r.t. mutexes) s-outgoing guards.
            s_guards = sorted(s_guards, key=lambda g: -len(self.__get_mutexes(g)))
            for guard in s_guards:
                if not self.__is_self_loop_id(guard):
                    # These are the rules with guard in the head. These should be treated disjunctively.
                    rules = list(filter(lambda rule: rule.head == guard, self.rules))
                else:
                    pass


class And:
    """Represents a conjunction of atoms. If negated = False, then to satisfy the conjunction
       all atoms need to be satisfied. If negated = True"""

    def __init__(self, atoms: list, negated: bool):
        if isinstance(atoms, And):
            pass
        self.conditions = atoms
        self.negated = negated


class Rule:
    def __init__(self, rule_id: str, head: str, body: list[str], mutexes: list[str] = None):
        self.rule_id = rule_id
        self.head = head
        self.body = body
        if mutexes is not None:
            self.mutexes = mutexes


class GuardBodyAtom:
    def __init__(self, clingo_atom):
        self.guard_head = str(clingo_atom.arguments[0])
        self.body_id = str(clingo_atom.arguments[1])
        self.body_atom = str(clingo_atom.arguments[2])
        self.str = str(clingo_atom)


class TransitionAtom:
    def __init__(self, clingo_atom):
        self.from_state = str(clingo_atom.arguments[0])
        self.guard = str(clingo_atom.arguments[1])
        self.to_state = str(clingo_atom.arguments[2])
        self.str = str(clingo_atom)


class SeqAtom:
    def __init__(self, clingo_atom):
        self.seq_id = int(str(clingo_atom.arguments[0]))
        self.attribute = str(clingo_atom.arguments[1].name)
        self.att_value = str(clingo_atom.arguments[1].arguments[0])
        self.time = int(str(clingo_atom.arguments[2]))
        self.str = str(clingo_atom)


class ClassAtom:
    def __init__(self, clingo_atom):
        self.seq_id = int(str(clingo_atom.arguments[0]))
        self.class_id = str(clingo_atom.arguments[1])
        self.str = str(clingo_atom)


class AcceptingStateAtom:
    def __init__(self, clingo_atom):
        self.state = str(clingo_atom.arguments[0])
        self.str = str(clingo_atom)


class CountsAtom:
    def __init__(self, clingo_atom):
        self.str = str(clingo_atom)


class AnyAtom:
    def __init__(self, clingo_atom):
        self.str = str(clingo_atom)


class MultiVarSeq:
    def __init__(self, seq_id):
        self.seq_id = seq_id
        self.sequences = {}
        self.class_id = None

    def sort_seqs(self):
        for k, v in self.sequences.items():
            v1 = sorted(v, key=lambda x: x[0])
            self.sequences[k] = v1

    def set_class(self, class_id):
        self.class_id = class_id


class EventTuple:
    def __init__(self, attr_val_pairs: list[str]):
        self.av_dict = {}
        for p in attr_val_pairs:
            s = p.split('=')
            attribute = s[0].strip()
            value = s[1].strip()
            self.av_dict[attribute] = value


class Guard:
    def __init__(self, guard_id: str, sfa: Automaton):
        s = guard_id.split(',')
        self.from_state = s[0].split('(')[1]
        self.to_state = s[1].split(')')[0]
        self.is_self_loop_guard = self.from_state == self.to_state


class GuardedTransition:
    """Used in the procedural (no ASP) evaluation of a model."""

    def __init__(self, atom: TransitionAtom, sfa: Automaton):
        self.from_state = atom.from_state
        self.to_state = atom.to_state
        self.guard = atom.guard
        self.automaton = sfa

        """The predicates that should evaluate to true for the transition to take place."""
        self.positive_conditions = []

        """The predicates that should evaluate to false for the transition to take place 
        (mutual exclusivity conditions in the case of DSFA)."""
        self.negative_conditions = []

    def get_conditions(self, guard):
        parsed = parse_term(guard)
        from_state, to_state = parsed.arguments[0], parsed.arguments[1]
        is_self_loop_guard = from_state == to_state
        if is_self_loop_guard:
            mutexes = self.automaton.self_loop_guards[guard]
            for guard in mutexes:  # These are always regular guards, no self-loops.
                conditions = self.automaton.rules_mutex_conditions[guard]
                guard_mutexes = self.automaton.rules_mutex_conditions[guard]
        else:
            pass


class SolveResult:
    def __init__(self, models, grounding_time, solving_time):
        self.models = models
        self.grounding_time = grounding_time
        self.solving_time = solving_time


class ScoredModel:

    def __init__(self,
                 automaton: Automaton,
                 constraints: list[str],
                 global_counts: (int, int, int),
                 scores_per_batch: dict):

        self.automaton = automaton
        self.constraints = constraints
        self.global_tps, self.global_fps, self.global_fns = \
            global_counts[0], global_counts[1], global_counts[2]
        self.scores_per_batch = scores_per_batch

    def global_f1(self):
        return f1(self.global_tps, self.global_fps, self.global_fns)

    def global_precision(self):
        return precision(self.global_tps, self.global_fps)

    def global_recall(self):
        return recall(self.global_tps, self.global_fns)

    def get_most_urgent_mini_batch(self, min_what, random_selection=True):
        """Get a mini batch where the current model performs poorly."""
        if min_what == 'precision':
            mini_batch = min(self.scores_per_batch.items(),
                             key=lambda item: precision(item[1][0], item[1][1]))

        elif min_what == 'recall':
            mini_batch = min(self.scores_per_batch.items(),
                             key=lambda item: recall(item[1][0], item[1][2]))

        else:  # F1-score
            get_f1 = lambda item: f1(item[1][0], item[1][1], item[1][2])

            # Remove any batch where the model makes no predictions
            useful = list(filter(lambda item: (item[1][0], item[1][1], item[1][2]) != (0, 0, 0),
                                 self.scores_per_batch.items()))

            if not random_selection:
                mini_batch = min(useful, key=lambda item: get_f1(item))
            else:
                non_perfect = list(filter(lambda item: get_f1(item) != 1.0, useful))
                if non_perfect:
                    # randomly select a mini-batch
                    mini_batch = random.choice(non_perfect)
                else:
                    # the model is perfect in the entire training set. Just return the first element.
                    # The caller method should terminate the MCTS run.
                    mini_batch = useful[0]

        if mini_batch[1][1] == 0 and mini_batch[1][2] == 0:
            stop = 'stop'

        mini_batch_id = mini_batch[0]
        return mini_batch_id


class TestResult:

    def __init__(self, tp_seqs, fp_seqs, fn_seqs, scored_paths, scores_per_batch):
        self.tp_seqs = tp_seqs
        self.fp_seqs = fp_seqs
        self.fn_seqs = fn_seqs
        self.scored_paths = scored_paths
        self.scores_per_batch = scores_per_batch

    def get_tps(self):
        if isinstance(self.tp_seqs, list):
            return len(self.tp_seqs)
        else:  # the actual counts
            return self.tp_seqs

    def get_fps(self):
        if isinstance(self.fp_seqs, list):
            return len(self.fp_seqs)
        else:  # the actual counts
            return self.fp_seqs

    def get_fns(self):
        if isinstance(self.fn_seqs, list):
            return len(self.fn_seqs)
        else:  # the actual counts
            return self.fn_seqs
