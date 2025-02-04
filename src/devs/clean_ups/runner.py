from src.asal.template import Template
from learntools import process_mini_batch
from src.asal.structs import Automaton
from src.asal.auxils import *

"""
==================================================================================================
This will crash due to Learner.induce_models() returning a SolveResult (that's how mcts.py works).
To get this to work just comment the current return statement there and uncomment 'return models'.
================================================================================================== 
"""

if __name__ == "__main__":
    # import subprocess subprocess.run(["clingo /home/nkatz/Dropbox/ASP-devs/automata/learn-conjunctions
    # -ndet.lp -t12 --time-limit=120"], shell=True)

    t_lim = 120  # 'inf'
    max_states = 6
    target_class = 1
    mini_batch_size = 20
    max_local_iters = 10
    min_precision = 0.95
    select_next_batch_by = "f1"
    time_limit = float('inf') if t_lim == 'inf' else t_lim
    template = Template(max_states, target_class)

    # train_path = '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TRAIN_SAX_8_ASP.csv'
    # test_path = '/home/nkatz/dev/TS-maritime_20200317/folds/fold_1/Maritime_TEST_SAX_8_ASP.csv'

    train_path = 'data-debug/Maritime_TRAIN_SAX_8_ASP.csv'
    test_path = 'data-debug/Maritime_TEST_SAX_8_ASP.csv'

    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=False)

    global_model, constraints = Automaton(), []
    best_performance = 0.0
    _continue = best_performance < 0.98
    selected_mini_batch = 1  # 47 with batch size 50
    debug = False
    multiproc = True

    while _continue:
        logger.debug(f'Selected mini-batch: {selected_mini_batch}')

        batch_data = train_data[selected_mini_batch]

        # constraints contain all accumulated constraints so far
        new_model, constrs = process_mini_batch(global_model, template, batch_data,
                                                mini_batch_size, train_path, t_lim,
                                                target_class, max_local_iters, min_precision,
                                                constraints, multiproc=multiproc, debug=debug)

        show_model_msg(new_model)

        selected_mini_batch = new_model.get_most_urgent_mini_batch(select_next_batch_by)

        if new_model.global_f1() > best_performance:
            best_performance = new_model.global_f1()
            global_model = new_model.automaton

        # constraints = constrs  # There is no reason to accumulate the constraints (I guess...)

