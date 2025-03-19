import math
import sys
from itertools import islice
import random
import pickle
from src.asal.logger import *
import time
import re


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if func.__name__ == "on_model":
            logger.debug(f"{func.__name__} took {end - start:.4f} seconds")
        if func.__name__ == "compile_sfa":
            logger.info(yellow(f'Compilation time: {end - start:.4f} secs'))
        if func.__name__ == "ground":
            logger.info(yellow(f'Grounding time: {end - start:.4f} secs'))
        if func.__name__ == "solve":
            logger.info(yellow(f'Solving time: {end - start:.4f} secs'))
        return result
    return wrapper

def sliding_window(elements, window_size):
    slices = []
    if len(elements) <= window_size:
        return [elements]

    for i in range(len(elements) - window_size + 1):
        slices.append((elements[i:i + window_size]))

    return slices


def get_path_cycles(state_seq: list):
    graph = get_graph_from_state_seq(state_seq)
    return get_cycles_dfs(graph)


def get_graph_from_state_seq(state_seq: list):
    """The input is a sequence of states generated from a run through an automaton. For instance:
       [1,3,1,3,1,3,1,3,1,3,5]
       meaning that the automaton starts from state 1, cycles over 1 and 3 for a number of steps
       and ends at state 5 (the final state).
       From this input this method generates a graph in the form of a dictionary where the keys are
       the states in the input and the values are each key's children. For instance, the output for
       the input above will be {'1': ['3', '5'], '3': ['1'], '5': []}
    """
    graph = {}
    last_seen = state_seq[0]
    graph[last_seen] = []
    state_seq.pop(0)
    for s in state_seq:
        if s not in graph.keys():
            graph[s] = []
        if s not in graph[last_seen]:
            graph[last_seen].append(s)
        last_seen = s
    return graph


def get_cycles_dfs(graph):
    """Find all cycles in a graph via depth-first search. The graph (e.g. automaton) is represented by
    a dictionary, where the keys are the node in the graph and each key's value is the list of its
    children nodes, e.g. graph = {1: [2, 3, 5], 2: [1], 3: [1], 4: [2], 5: [2]}"""
    cycles = [[node] + path for node in graph for path in dfs(graph, node, node)]
    return cycles


def dfs(graph, start, end):
    """A DFS alg. for extracting all cycles from a graph."""
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path + [next_state]))


def flatten(_list):
    return [item for sublist in _list for item in sublist]


def get_pickled_fsm(path: str):
    """
    Returns one of the pickled automata in the /debug folder. The path is something like:
    debug/pickled_fsms/rules-size-3.
    Variables in the FSM can be accessed like:
    print(fsm.rules)
    print(fsm.self_loop_guards)
    print(fsm.to_string)
    """
    file = open(path, 'rb')
    fsm = pickle.load(file)
    file.close()
    return fsm


def split_data_rev(train_path: str, target_class: str, mini_batch_size: int, shuffle=False):
    """
    Used for incremental learning.
    Splits the training data into mini-batches of positive and negative examples.
    This method expects that the data is in ASP format. An example may be found in:
    /home/nkatz/dev/datasets_asp_wayeb_04062021/BioSmall/folds/fold_0/MTS_TRAIN_SAX_8_ASP.csv
    Sequences that are part of the same example are identified via their id. For instance,
    the following sequences represent two examples with ids 21 and 22 respectively. The first
    is a positive example and the second is a negative one.
    seq(21,alive(e),0). seq(21,alive(e),1). seq(21,alive(e),2). class(21,1).
    seq(21,apoptotic(b),0). seq(21,apoptotic(b),1). seq(21,apoptotic(b),2). class(21,1).
    seq(21,necrotic(a),0). seq(21,necrotic(a),1). seq(21,necrotic(b),2). class(21,1).
    seq(22,alive(e),0). seq(22,alive(e),1). seq(22,alive(e),2). class(22,0).
    seq(22,apoptotic(b),0). seq(22,apoptotic(b),1). seq(22,apoptotic(b),2). class(22,0).
    seq(22,necrotic(a),0). seq(22,necrotic(a),1). seq(22,necrotic(a),2). class(22,0).

    This method does the following:
    1. Separates positive/negative examples in two dictionaries where a key is the id of a sequence
       and the value is a list of strings, each representing a sequence in the example id (multi-variate case).
    2. Generates mini-batches by mixing positive/negative examples to create a mini-batch.
    :param train_path: The path to the training data.
    :param target_class: The class treated as positive to separate examples.
    :param mini_batch_size: ...
    :return: A list of lists (mini-batches)
    """

    def exmple_info(string):
        """Helper function to find if an example is positive or negative and to extract its id."""
        # class_atom = string.split(" ").pop()
        # print(f'string is {string}')
        class_predicate_match = re.search(r'class\(\d+,\s*\d+\)', string)
        if class_predicate_match:
            class_atom = class_predicate_match.group()
            split = class_atom.split(",")
            class_value = split[1].split(")")[0].strip()
            exmpl_id = split[0].split("(")[1].strip()
            return exmpl_id, class_value
        else:
            raise RuntimeError('Class atom not found.')

    def chunk_list(data, chunk_size):
        it = iter(data)
        res = []
        for i in range(0, len(data), chunk_size):
            # yield {k: data[k] for k in islice(it, chunk_size)}
            res.append({k: data[k] for k in islice(it, chunk_size)})
        return res

    def shuffle_dict(d: dict):
        keys = list(d.keys())
        random.shuffle(keys)
        return dict([(key, d[key]) for key in keys])

    _pos, _neg = {}, {}

    with open(train_path) as training_data:
        lines = training_data.readlines()
        for seq in lines:
            if seq.strip() != '' and seq.strip() != '\n':
                info = exmple_info(seq)
                exmpl_id, label = info[0], info[1]
                store = _pos if label == str(target_class) else _neg
                if exmpl_id in store:
                    store[exmpl_id].append(seq)
                else:
                    store[exmpl_id] = [seq]

    exmpl_count = len(_pos) + len(_neg)
    mini_batch_count = float(exmpl_count) / mini_batch_size

    # Find the number of positive/negative examples that a mini-batch should contain.
    pos_per_batch = int(math.ceil(len(_pos) / float(mini_batch_count)))
    neg_per_batch = int(math.ceil(len(_neg) / float(mini_batch_count)))

    if shuffle:
        _pos = shuffle_dict(_pos)
        _neg = shuffle_dict(_neg)

    # The following are lists of dictionaries. Each such dictionary contains
    # approximately the proper number of examples (positive or negative) that
    # a mini-batch should contain.
    if _pos == {}:
        logger.error('There might be a problem with th class label. No positive examples found!')
        sys.exit()
    pos_chunked = chunk_list(_pos, pos_per_batch)
    neg_chunked = chunk_list(_neg, neg_per_batch) if _neg else []

    # Zip positive and negatives together, to get mini-batches that contain both types of
    # examples. The ratio of positive/negative examples in each mini-batch follows the one
    # in the entire training set. After zipping, the examples in the "tail" of the largest
    # list are not paired with examples of the opposite type. For instance if we have 40
    # mini-batches with 2 positive in each and 50 mini-batches with 10 negative examples in each
    # after the zip operation we'll have 40 mini-batches with 12 examples in each and 10
    # "leftover" mini-batches with 10 negative examples in each one.
    if _neg:  # Standard case, where we have both positive and negative seqs.
        largest = pos_chunked if len(pos_chunked) >= len(neg_chunked) else neg_chunked
        difference = abs(len(pos_chunked) - len(neg_chunked))
        tail = largest[-difference:] if difference > 0 else []  # Keep the leftovers
        zipped = list(zip(pos_chunked, neg_chunked))

        # Zipped consists of pairs (2-tuples) of mini-batches: (x,y), where x is a mini-batch of positive
        # and y is a mini-batch of negative examples (note that x,y are still dictionaries). We merge
        # these dicts together for each entry in zipped:
        mini_batches = [{**x, **y} for (x, y) in zipped]
        for i in tail:
            mini_batches.append(i)  # We also add the leftovers to the list of mini_batches

        return mini_batches
    else:  # No negatives, for a setting where we are learning a fixed-size automaton that accepts all input seqs.
        return pos_chunked


def extract_data(mini_batch: dict[int: list[str]]):
    """
    :param mini_batch: A dict[int: list[str]] where the key is the id of the example and the value is the list
                       of the sequences of that example (more than one seqs in case of multi-variate examples).
    :return: A string representation of the data to pass to Clingo.
    """
    flat_list = [seq for seq_list in mini_batch.values() for seq in seq_list]
    return "\n".join(flat_list)


def get_train_data(train_data_path: str, target_class: str, mini_batch_size: int, shuffle=False):
    start = time.time()
    mini_batches = split_data_rev(train_data_path, target_class, mini_batch_size, shuffle)
    mini_batch_dict = {}
    for i in range(len(mini_batches)):
        mini_batch_dict[i] = extract_data(mini_batches[i])
    end = time.time()
    logger.info(f'Get data time: {end - start} sec')
    return mini_batch_dict


def split_by_n(my_list: list, n: int):
    """split my_list into chunks of size n"""
    return [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]


def get_seqs_by_id(seq_id_list: list, training_path: str):
    """Fetches data sequences by id, for all ids in seq_id_list"""
    prefix_list = list(map(lambda _id: f'seq({_id}', seq_id_list))
    test = lambda string: list(filter(string.startswith, prefix_list)) != []
    with open(training_path) as training_data:
        lines = training_data.readlines()
        filter_out = [line for line in lines if test(line)]
    result = '\n'.join(filter_out)
    return result


def precision(tps, fps):
    return float(tps) / (tps + fps) if tps + fps > 0 else 0.0


def recall(tps, fns):
    return float(tps) / (tps + fns) if tps + fns > 0 else 0.0


def f1(tps, fps, fns):
    p = precision(tps, fps)
    r = recall(tps, fns)
    return 2 * p * r / (p + r) if p + r > 0.0 else 0.0


def f1_aux(counts):
    tps, fps, fns = counts[0], counts[1], counts[2]
    p = precision(tps, fps)
    r = recall(tps, fns)
    return 2 * p * r / (p + r) if p + r > 0.0 else 0.0


def DISC_to_ASP_DFAs(dfa):
    transitions = dfa.split("\n")
    states, symbols, asp_atoms = set(()), set(()), set(())
    for t in transitions:
        s = t.split(" = ")
        to_state = s[1]
        ss = s[0].split(",")
        from_state = ss[0].split("(")[1]
        symbol = ss[1].split(")")[0]
        states.update([from_state, to_state])
        symbols.add(symbol)
        asp_atom = "transition({0},{1},{2})".format(from_state, symbol, to_state)
        asp_atoms.add(asp_atom)
    acc_atoms = ["accepting({0})".format(x) for x in states if (int(x) % 2 == 0 and x != '0')]
    asp_atoms.update(acc_atoms)
    return asp_atoms


def show_model_msg(new_model):
    show = green(new_model.automaton.show(mode='reasoning'))
    tps = new_model.global_tps
    fps = new_model.global_fps
    fns = new_model.global_fns
    f_1 = new_model.global_f1()
    msg = green(f'\nBest model so far (training set TPs, FPs, FNs, F1: {tps}, {fps}, {fns}, {f_1})', 'underlined')
    logger.debug(f'{msg}\n{show}\n')
