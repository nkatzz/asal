import multiprocessing
import clingo
from clingo.script import enable_python
from src.asal.structs import Automaton
from itertools import groupby
from src.asal.asp import get_test_program


class Tester:

    def __init__(self, data, args, automaton, path_scoring=False):

        domain, target_class = args.domain, args.tclass
        self.args = args
        self.cores = multiprocessing.cpu_count()
        self.data = data
        self.model = automaton
        self.target_class = target_class
        self.path_scoring = path_scoring
        self.test_program = get_test_program(args)

        cores_num = multiprocessing.cpu_count()
        # cores = f'-t{cores_num if cores_num <= 64 else 64}'
        # self.ctl = clingo.Control([cores])  # clingo.Control([cores, "--warn=none"])  "--warn=no-atom-undefined"
        if args.warns_off:
            self.ctl = clingo.Control(["--warn=none"])
        else:
            self.ctl = clingo.Control()

        # key: a path in the form '1->3->5' (the last state is accepting). Value: tp, fp counts for that path,
        # so that we can calculate path precision. FNs are calculated w.r.t. the entire automaton.
        self.scored_paths: dict[str, (int, int)] = {}
        self.tp_seq_ids = []
        self.fp_seq_ids = []
        self.fn_seq_ids = []

        self.tp_earliness = {}
        self.fp_earliness = {}

    def count_stats(self, model):
        """Used when we are just evaluating an automaton on the training/testing set.
           We are simply counting stats here."""
        for atom in model.symbols(shown=True):
            seq_id = str(atom.arguments[0])
            if atom.match("tp", 1):
                self.tp_seq_ids.append(seq_id)
            elif atom.match("fp", 1):
                self.fp_seq_ids.append(seq_id)
            elif atom.match("fn", 1):
                self.fn_seq_ids.append(seq_id)
            else:
                pass

    def count_stats_with_earliness(self, model):
        # tps_seqs & fp_seqs are dicts of the form {seq_id: earliness_val}, fn_seqs is a list with fn seq ids.
        tps_seqs, fp_seqs, fn_seqs = {}, {}, []
        for atom in model.symbols(shown=True):
            seq_id = str(atom.arguments[0])
            if atom.match("fn", 1):
                self.fn_seq_ids.append(seq_id)
            else:
                accept_time, seq_len = int(str(atom.arguments[1])), int(str(atom.arguments[2]))
                earliness = (seq_len - accept_time) / seq_len
                if atom.match("tp", 3):
                    self.tp_earliness[seq_id] = earliness
                else:
                    self.fp_earliness[seq_id] = earliness

    def calculate_earliness(self):
        def prf(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            return precision, recall, f1

        tps, fps, fns = len(self.tp_earliness), len(self.fp_earliness), len(self.fn_seq_ids)
        avg_tp_earliness, avg_fp_earliness = None, None
        if tps > 0:
            avg_tp_earliness = sum(self.tp_earliness.values()) * 100 / tps
        if fps > 0:
            avg_fp_earliness = sum(self.fp_earliness.values()) * 100 / fps
        p, r, f1 = prf(tps, fps, fns)
        print(f'F1-score: {f1:.3f} (tps, fps, fns: {tps}, {fps}, {fns})')
        if avg_tp_earliness is not None:
            print(f'Average tp earliness: {avg_tp_earliness:.0f}%')
        if avg_fp_earliness is not None:
            print(f'Average fp earliness: {avg_fp_earliness:.0f}%')

    @staticmethod
    def __split(data, predicate):
        """Helper method for doing what the partition function does in Scala."""
        s1, s2 = [], []
        for d in data:
            if predicate(d):
                s1.append(d)
            else:
                s2.append(d)
        return s1, s2

    @staticmethod
    def __find(data, predicate):
        """Helper method for doing what the find function does in Scala."""
        s1 = []
        for d in data:
            if predicate(d):
                s1.append(d)
        return s1

    def score_paths(self, model):
        """Used when we need to collect the necessary statistics to calculate precision per path
           in order to repeat a local search with bad paths as constraints."""

        def sequence_id(a):
            return str(a.arguments[0])

        def from_state(a):
            return str(a.arguments[1])

        def to_state(a):
            return str(a.arguments[2])

        def time(a):
            return int(str(a.arguments[3]))

        def guard_id(a):
            return str(a.arguments[4])

        # The model here consists of accepting_path_edge/5 atoms and tp/1, fp/1, fn/1 atoms. First, split them.
        path_predicates, stat_predicates = self.__split(model.symbols(shown=True),
                                                        lambda a: a.match("accepting_path_edge", 5))
        stats = {}
        for atom in stat_predicates:
            s = str(atom).split('(')
            funct = s[0]
            seq_id = s[1].split(')')[0]
            stats[seq_id] = funct
            # Update the FN counts (these are not attributed to paths naturally)
            if funct == 'fn':
                self.fn_seq_ids.append(seq_id)

        # x = self.__find(path_predicates, lambda a: sequence_id(a) == '19')

        """Group the path_predicates by SeqIq. Important note: with itertools groupby only consecutive elements of  
        the same value are grouped. To group them regardless of order, we need to sort the original list first."""
        path_edges_per_seq = [list(group) for _, group in
                              groupby(sorted(path_predicates, key=lambda p: int(sequence_id(p))),
                                      lambda q: sequence_id(q))]

        for seq_edges_group in path_edges_per_seq:
            path = ['1']
            seq_edges_group.sort(key=lambda a: (time(a)))  # starts from state 1 and all other edge atoms follow.
            seq_id = sequence_id(seq_edges_group[0])
            for atom in seq_edges_group:
                to = to_state(atom)
                guard = guard_id(atom)
                _from = path.pop()
                path.append((_from, guard))
                path.append(to)
            path_as_key = '->'.join(map(lambda a: str(a), path))
            seq_eval = stats[seq_id]

            if path_as_key in self.scored_paths.keys():
                old = self.scored_paths[path_as_key]
                if seq_eval == 'tp':
                    new = (old[0] + 1, old[1])
                    self.tp_seq_ids.append(seq_id)
                else:
                    new = (old[0], old[1] + 1)
                    self.fp_seq_ids.append(seq_id)
                self.scored_paths[path_as_key] = new
            else:
                if seq_eval == 'tp':
                    self.scored_paths[path_as_key] = (1, 0)
                    self.tp_seq_ids.append(seq_id)
                else:
                    self.scored_paths[path_as_key] = (0, 1)
                    self.fp_seq_ids.append(seq_id)

    def test_model(self, test_data_from_file=False):
        enable_python()
        """
        if 'src' in os.getcwd():
            path_scoring_file = os.path.normpath(
                os.getcwd() + os.sep + 'asp' + os.sep + 'path_scoring.lp')
            test_model_file = os.path.normpath(
                os.getcwd() + os.sep + 'asp' + os.sep + 'test_model.lp')
        else:
            path_scoring_file = os.path.normpath(
                os.getcwd() + os.sep + 'src' + os.sep + 'asp' + os.sep + 'path_scoring.lp')
            test_model_file = os.path.normpath(
                os.getcwd() + os.sep + 'src' + os.sep + 'asp' + os.sep + 'test_model.lp')

        infer_file = path_scoring_file if self.path_scoring else test_model_file
        # infer_file = 'asp/path_scoring.lp' if self.path_scoring else 'asp/test_model.lp'
        ctl.load(infer_file)
        if test_data_from_file:
            ctl.load(self.data)
        else:
            ctl.add("base", [], self.data)
        """

        self.ctl.add("base", [], self.test_program)
        self.ctl.add("base", [], self.data)
        # self.ctl.add("base", [], f'targetClass({self.target_class}).')

        if isinstance(self.model, Automaton):
            self.ctl.add("base", [], self.model.show(mode='reasoning'))
            self.ctl.add("base", [], self.model.accepting_states[0])
        else:  # We pass an automaton in string format to test it.
            self.ctl.add("base", [], self.model)

        if self.path_scoring:
            # Add Some extra necessary info for performing the scoring.
            start = 'start(1).'
            target_class = f'targetClass({self.target_class}).'
            self.ctl.add("base", [], start)
            self.ctl.add("base", [], target_class)

        self.ctl.ground([("base", [])])
        if self.path_scoring:
            self.ctl.solve(on_model=self.score_paths)
        else:
            if 'eval_earliness' in self.args and self.args.eval_earliness is not None:
                self.ctl.solve(on_model=self.count_stats_with_earliness)
            else:
                self.ctl.solve(on_model=self.count_stats)


def rewrite_automaton(automaton: Automaton):
    """
    Rewrites an automaton into a form that allows to keep track of the particular rules that
    trigger transitions during path scoring. This is necessary because disjunctive guards definitions may involve
    multiple rules with the same head and each such rule may be involved in a different accepting path.
    For instance, the automaton below contains a disjunctive definition for f(1,5).

    transition(1,f(1,1),1). transition(1,f(1,5),5). transition(1,f(1,3),3). transition(3,f(3,5),5).
    transition(5,f(3,3),5). transition(5,f(5,5),5). accepting(5).
    holds(f(1,5),S,T) :- holds(at_least(heading,c),S,T).
    holds(f(1,5),S,T) :- holds(at_most(lon,f),S,T), holds(at_most(lat,g),S,T).
    holds(f(1,3),S,T) :- holds(at_most(speed,b),S,T).
    holds(f(3,5),S,T) :- holds(at_most(speed,c),S,T).
    holds(f(5,5),S,T) :- sequence(S), time(T).
    holds(f(3,3),S,T) :- sequence(S), time(T), not holds(f(3,5),S,T).
    holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,5),S,T), not holds(f(1,3),S,T).

    This automaton is transformed as follows. First, give a different
     id to all the rules, as in the following dictionary:

    {
    1: (f(1,5), [holds(at_least(heading,c),S,T)])
    2: (f(1,5), [holds(at_most(lon,f),S,T), holds(at_most(lat,g),S,T)])
    3: (f(1,3), [holds(at_most(speed,b),S,T)])
    4: (f(3,5), [holds(at_most(speed,c),S,T)])
    5: (f(5,5), [])
    6: (f(3,3), [f(3,5)])
    7: (f(1,1), [f(1,5), f(1,3)])
    }

    Next, the initial rules are re-written, using a new id derived by their corresponding key in the dictionary:

    holds(f(1),S,T) :- holds(at_least(heading,c),S,T).
    holds(f(2),S,T) :- holds(at_most(lon,f),S,T), holds(at_most(lat,g),S,T).
    holds(f(3),S,T) :- holds(at_most(speed,b),S,T).
    holds(f(4),S,T) :- holds(at_most(speed,c),S,T).
    holds(f(5),S,T) :- sequence(S), time(T).
    holds(f(6),S,T) :- sequence(S), time(T), not holds(f(4),S,T).
    holds(f(7),S,T) :- sequence(S), time(T), not holds(f(1),S,T), not holds(f(2),S,T), not holds(f(3),S,T).

    Finally, the structure of the automaton is transformed in the same way:

    transition(1,f(7),1). transition(1,f(1),5). transition(1,f(2),5). transition(1,f(3),3). transition(3,f(4),5).
    transition(3,f(6),3). transition(5,f(5),5). accepting(5).
    """

    new_fsm_map, self_loop_heads, new_rules, new_transitions, count = {}, [], [], [], 1

    """Transform the regular rules in the automaton."""

    for rule in automaton.rules:
        """Create the map first and then re-iterate over the rules to generate their new versions. 
                This is necessary in order to properly map mutual exclusivity conditions from the old rules
                to the new ones."""
        new_fsm_map[count] = (rule.head, rule.body)
        count += 1

    for k, v in new_fsm_map.items():
        new_id, old_id, body = k, v[0], v[1]
        mutexes = automaton.rules_mutex_conditions[old_id]
        new_mutexes = []
        for m in mutexes:
            for k1, v1 in new_fsm_map.items():
                if v1[0] == m:
                    new_mutexes.append(f'f({k1})')
        new_head = f'holds(f({new_id}),S,T)'
        _new_body = list(map(lambda x: f'holds({x},S,T)', body))
        mutes = list(map(lambda x: f'not holds({x},S,T)', new_mutexes))

        new_body = ', '.join(_new_body + mutes)
        new_rule = f'{new_head} :- {new_body}.'
        new_rules.append(new_rule)

    """Transform the self-loop rules."""
    for k, v in automaton.self_loop_guards.items():
        self_loop_heads.append(k)
        new_fsm_map[count] = (k, v)
        """We need to find the new f(i)'s that correspond to the f(i,j)'s which are negated in the bodies
        of self-loop rules. Note that if we have a disjunctive definition for some f(i,j), then we need 
        to add to the body of the transformed self-loop rule a negated atom for each alternative definition
        of that f(i,j). For example, assume that f(2,3) is defined by two rules, which correspond to e.g
        f(5) and f(6) in the new ids. Then the self-loop rule f(2,2), assumming that it corresponds to e.g.
        f(8) in the new ids, should be transformed to f(8) :- not f(5), not f(6) (while originally it 
        corresponds to f(2,2) :- not f(2,3)). Concrete example: consider the automaton:

        accepting(5).
        transition(1,f(1,1),1). transition(1,f(1,4),4). transition(4,f(4,4),4). 
        transition(4,f(4,5),5). transition(5,f(5,5),5).
        holds(f(1,4),S,T) :- holds(at_most(latitude,c),S,T), holds(at_least(latitude,d),S,T).
        holds(f(1,4),S,T) :- holds(at_most(latitude,e),S,T), holds(at_most(speed,a),S,T).
        holds(f(4,5),S,T) :- holds(at_least(latitude,c),S,T), holds(at_least(speed,g),S,T).
        holds(f(4,5),S,T) :- holds(at_least(latitude,d),S,T).
        holds(f(1,1),S,T) :- sequence(S), time(T), not holds(f(1,4),S,T).
        holds(f(4,4),S,T) :- sequence(S), time(T), not holds(f(4,5),S,T).
        holds(f(5,5),S,T) :- sequence(S), time(T).

        The new_fsm_map for this automaton is:

        {
         1: ('f(1,4)', ['at_most(latitude,c)', 'at_least(latitude,d)']), 
         2: ('f(1,4)', ['at_most(latitude,e)', 'at_most(speed,a)']), 
         3: ('f(4,5)', ['at_least(latitude,c)', 'at_least(speed,g)']), 
         4: ('f(4,5)', ['at_least(latitude,d)']), 
         5: ('f(1,1)', ['f(1,4)']), 
         6: ('f(4,4)', ['f(4,5)']), 
         7: ('f(5,5)', [])
        }

        The new_rules with the new representation are:

        [
         'holds(f(1),S,T) :- holds(at_most(latitude,c),S,T), holds(at_least(latitude,d),S,T).', 
         'holds(f(2),S,T) :- holds(at_most(latitude,e),S,T), holds(at_most(speed,a),S,T).', 
         'holds(f(3),S,T) :- holds(at_least(latitude,c),S,T), holds(at_least(speed,g),S,T).', 
         'holds(f(4),S,T) :- holds(at_least(latitude,d),S,T).', 
         'holds(f(5),S,T) :- not holds(f(1),S,T), not holds(f(2),S,T).', 
         'holds(f(6),S,T) :- not holds(f(3),S,T), not holds(f(4),S,T).', 
         'holds(f(7),S,T) :- sequence(S), time(T).'
        ]
        """
        mutexes = automaton.self_loop_guards[k]
        new_mutexes = []
        for m in mutexes:
            for k1, v1 in new_fsm_map.items():
                if v1[0] == m:
                    new_mutexes.append(f'f({k1})')

        new_head = f'holds(f({count}),S,T)'
        types = ['sequence(S)', 'time(T)']
        mutes = list(map(lambda x: f'not holds({x},S,T)', new_mutexes))
        new_body = ', '.join(types + mutes)
        new_rule = f'{new_head} :- {new_body}.'
        new_rules.append(new_rule)
        count += 1

    """Transform the transition facts."""
    for atom in automaton.transitions:
        from_state, guard, to_state = atom.from_state, atom.guard, atom.to_state
        new_guard_atoms = []
        for k, v in new_fsm_map.items():
            if v[0] == guard:
                new_guard_atoms.append(f'f({k})')
        for g in new_guard_atoms:
            new_transitions.append(f'transition({from_state},{g},{to_state}).')

    """
    print(new_fsm_map)
    print(new_rules)
    print(new_transitions)
    """

    """Return also a string representation of the automaton to use directly."""
    new_automaton_str = '\n'.join(automaton.accepting_states + new_transitions + new_rules)

    return new_automaton_str, new_fsm_map
