from src.asal.auxils import get_path_cycles, precision, sliding_window
from src.logger import *
from src.asal.structs import Automaton


def generate_constraints(scored_paths: dict[str, (int, int)],
                         guards_map: dict, min_precision, existing_model: Automaton):

    def allowed_to_constrain(guard_head, guard_body_atom):
        """Checks if a body/3 atom is one of those that the existing_model consists of.
        If that's the case, then we don't generate a constraint to exclude that atom from
        future solutions. This may happen when an otherwise high-quality rule participates
        in a low-quality path that has just been generated, due to the induction of a new
        (most likely low-quality) rule."""

        existing = lambda a: a.guard_head == guard_head and a.body_atom == guard_body_atom
        x = any([existing(atom) for atom in existing_model.body_atoms_asp])
        return not x

    bad_paths, constraints = [], []

    for path, score in scored_paths.items():

        print(path, score)
        path_to_list = path.split('->')
        accepting_state = path_to_list.pop(len(path_to_list) - 1)
        path_to_list = de_string_path(path_to_list)
        p = precision(score[0], score[1])
        if p < min_precision:
            bad_paths.append((path_to_list, p))

    if bad_paths:
        logger.debug(f'Bad paths:\n{show_bad_paths(bad_paths)}')

    for p in bad_paths:
        path, _precision = p[0], p[1]
        functs = [x[1] for x in path]
        for f in functs:
            guard_id = int(f.split('(')[1].split(')')[0])
            actual_guard_id, guard_body = guards_map[guard_id][0], guards_map[guard_id][1]

            """
            constraint_body = ', '.join([f'body({actual_guard_id},J,{atom})'
                                         for atom in guard_body if allowed_to_constrain(actual_guard_id, atom)])
            constraint = f':- {constraint_body}.' if constraint_body.strip() != '' else ''
            if constraint not in constraints and constraint.strip() != '':
                constraints.append(constraint)
            """

            # Generate ground constraints to avoid re-grounding
            for i in range(1, 4):
                constraint_body = ', '.join([f'body({actual_guard_id},{i},{atom})'
                                             for atom in guard_body if allowed_to_constrain(actual_guard_id, atom)])
                constraint = f':- {constraint_body}.' if constraint_body.strip() != '' else ''
                if constraint not in constraints and constraint.strip() != '':
                    constraints.append(constraint)

    if constraints:
        logger.debug(f'Generated constraints:\n{show_constraints(constraints)}')

    return constraints


def generate_path_constraints(scored_paths: dict[str, (int, int)],
                              guards_map: dict, min_precision):
    """Generates variabilized path constraints of the form e.g.
           :- holds(f(1,X1),S,T1), body(f(1,X1),J1,at_least(heading,g)), holds(f(X1,5),S,T2),
              body(f(X1,5),J2,at_most(heading,a)), T1 < T2.
        which are impossible to be reasoned upon. It's not even clear if these constraints are any useful."""
    bad_paths, constraints = [], []

    for path, score in scored_paths.items():
        pr = precision(score[0], score[1])
        # if pr < min_precision:

        print(path, score)
        path_to_list = path.split('->')
        accepting_state = path_to_list.pop(len(path_to_list) - 1)
        path_to_list = de_string_path(path_to_list)
        path_to_list.append((accepting_state, ''))
        p = precision(score[0], score[1])
        if p < min_precision:
            bad_paths.append((path_to_list, p))

    if bad_paths:
        logger.debug(f'Bad paths:\n{show_bad_paths(bad_paths)}')

    for p in bad_paths:
        path, _precision = p[0], p[1]
        final_state = path[len(path) - 1]
        path_copy = [x for x in path]  # cycle extraction involves a pop() that messes with the original list.
        cycles = get_path_cycles(path_copy)
        if not cycles:  # empty
            c = get_constraint(path, False, guards_map)
            constraints.append(c)
        else:
            for cycle in cycles:
                cycle_copy = [x for x in cycle]
                cycle_copy.append(final_state)
                c = get_constraint(cycle_copy, True, guards_map)
                constraints.append(c)

    if constraints:
        logger.debug(f'Generated constraints:\n{show_constraints(constraints)}')

    return constraints


def show_bad_paths(bad_paths):
    with_score = [' -> '.join(list(map(lambda x: str(x).replace("""\'""", ''), path))) + f': {score}' for path, score in
                  bad_paths]
    res = '\n'.join(with_score)
    return res


def show_constraints(constr):
    return '\n'.join(constr)


def _split(state_guard_tuple: str):
    s = state_guard_tuple.split(',')
    state = s[0].split("(")[1].replace('\'', '')
    guard = s[1].split(')')[0].replace('\'', '') + ')'
    return state, guard


def de_string_path(path_str: list[str]):
    de_stringed_list = [_split(x) for x in path_str]
    return de_stringed_list


def var_path(_path: list[(str, str)], is_cycle):
    _path_copy = [x for x in _path]
    final = _path_copy.pop(len(_path_copy) - 1)
    varbed = []
    if not is_cycle:
        start = _path_copy.pop(0)
        varbed.append(start)
        for i in range(0, len(_path_copy)):
            new_tuple = f'X{i + 1}', _path_copy[i][1]
            varbed.append(new_tuple)
        varbed.append(final)
    else:
        states_map = {}
        for t, i in zip(_path_copy, range(1, len(_path_copy) + 1)):
            if t[0] not in states_map.keys():
                states_map[t[0]] = f'X{i}'
        for t in _path_copy:
            new_tuple = states_map[t[0]], t[1]
            varbed.append(new_tuple)
        """we are simply excluding "bad" cycles, regardless of how they lead to an accepting state
        so comment the line below."""
        # varbed.append(final)
    return varbed


def get_constraint(_path, is_cycle, guards_map):
    """Generates variabilized path constraints of the form e.g.
       :- holds(f(1,X1),S,T1), body(f(1,X1),J1,at_least(heading,g)), holds(f(X1,5),S,T2),
          body(f(X1,5),J2,at_most(heading,a)), T1 < T2.
    which are impossible to be reasoned upon. It's not event clear if these constraints are any useful."""
    varbed = var_path(_path, is_cycle)
    _path_copy = [x for x in _path]
    transitions = sliding_window(varbed, 2)
    _transitions_ = [(t[0][0], t[1][0], t[0][1]) for t in transitions]
    constraint_body, time_vars = [], []

    for transition_tuple, time_var_count in zip(_transitions_, range(1, len(_transitions_) + 1)):
        from_state, to_state, guard = transition_tuple[0], transition_tuple[1], transition_tuple[2]
        guard_id = int(guard.split('(')[1].split(')')[0])
        actual_guard_id, guard_body = guards_map[guard_id][0], guards_map[guard_id][1]
        t_var, rule_body_var, guard_id = f'T{time_var_count}', f'J{time_var_count}', f'f({from_state},{to_state})'
        constraint_body.append(f'holds({guard_id},S,{t_var})')
        # body(f(1,3),J1,at_most(latitude,b))
        constraint_body = constraint_body + list(map(lambda x: f'body({guard_id},{rule_body_var},{x})', guard_body))
        time_vars.append(t_var)

    if len(time_vars) > 1:
        time_pairs = sliding_window(time_vars, 2)
        if len(time_pairs) > 1:
            time_sorting = [f'{x[0]} < {x[1]}' for x in time_pairs]
        else:
            x = time_pairs[0]
            time_sorting = [f'{x[0]} < {x[1]}']
        constraint = ':- ' + ', '.join(constraint_body) + ', ' + ', '.join(time_sorting) + '.'
    else:
        constraint = ':- ' + ', '.join(constraint_body) + '.'
    return constraint


if __name__ == "__main__":
    _scored_paths = {"('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(4)')->5": (144, 0),
                     "('1', 'f(7)')->5": (172, 149),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(3)')->5": (159, 1),
                     "('1', 'f(5)')->('4', 'f(3)')->5": (431, 385),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(4)')->5": (
                         41, 0), "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (136, 2),
                     "('1', 'f(6)')->('3', 'f(4)')->5": (252, 148),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         5, 1), "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(6)')->('3', 'f(4)')->5": (68, 1),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(4)')->5": (
                         26, 0),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(6)')->('3', 'f(1)')->('1', 'f(6)')->('3', 'f(4)')->5": (
                         4, 0),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         9, 1),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         6, 2),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(1)')->('1', 'f(6)')->('3', 'f(4)')->5": (
                         1, 0),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         1, 1),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         2, 0),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         11, 1),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         0, 1),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         1, 0),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         2, 0),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         3, 0),
                     "('1', 'f(6)')->('3', 'f(1)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         1, 0),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(6)')->('3', 'f(4)')->5": (
                         1, 0),
                     "('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(2)')->('1', 'f(5)')->('4', 'f(3)')->5": (
                         1, 0)}
    _guards_map = {1: ('f(3,1)',
                       ['at_most(longitude,f)', 'at_most(latitude,b)', 'at_most(speed,f)', 'at_most(heading,e)',
                        'at_least(heading,b)']), 2: ('f(4,1)',
                                                     ['at_most(longitude,f)', 'at_most(latitude,b)', 'at_most(speed,f)',
                                                      'at_most(heading,d)', 'at_least(heading,b)']),
                   3: ('f(4,5)', ['at_most(longitude,f)', 'at_least(latitude,c)']),
                   4: ('f(3,5)', ['at_most(latitude,e)', 'at_most(course_over_ground,b)', 'at_least(latitude,d)']), 5: (
            'f(1,4)', ['at_most(speed,f)', 'at_most(heading,c)', 'at_least(speed,d)', 'at_least(heading,b)']),
                   6: ('f(1,3)', ['at_most(speed,g)', 'at_most(course_over_ground,a)', 'at_least(latitude,b)']), 7: (
            'f(1,5)',
            ['at_most(speed,d)', 'at_most(heading,f)', 'at_least(speed,d)', 'at_least(course_over_ground,f)']),
                   8: ('f(1,1)', ['f(1,3)', 'f(1,4)', 'f(1,5)']), 9: ('f(5,5)', []),
                   10: ('f(3,3)', ['f(3,1)', 'f(3,5)']), 11: ('f(4,4)', ['f(4,1)', 'f(4,5)']), 12: ('f(2,2)', [])}

    constrs = generate_path_constraints(_scored_paths, _guards_map, 0.9)
    for x in constrs:
        print(x)
