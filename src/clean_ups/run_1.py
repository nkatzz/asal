from src.asal.template import Template
from src.asal.structs import Automaton
from src.asal.auxils import *
from src.asal.learner import Learner
from multiprocessing import Process, Queue


def call_learner(_learner: Learner, q: Queue,
                 _train_path, _target_class,
                 _mini_batch_size, _min_precision,
                 _global_model, _max_local_iters):

    q.put(_learner.induce_iteratively(_train_path, _target_class,
                                      _mini_batch_size, _min_precision,
                                      _global_model, _max_local_iters))


def call(_learner: Learner, _train_path,
         _target_class, _mini_batch_size,
         _min_precision, _global_model, _max_local_iters):

    q = Queue()
    process = Process(target=call_learner, args=(_learner, q, _train_path,
                                                 _target_class, _mini_batch_size,
                                                 _min_precision, _global_model, _max_local_iters))
    process.start()
    model = q.get()
    process.join()
    return model


if __name__ == "__main__":

    t_lim = 120  # 'inf'
    max_states = 6
    target_class = 1
    mini_batch_size = 20
    max_local_iters = 10
    min_precision = 0.95
    select_next_batch_by = "f1"
    time_limit = float('inf') if t_lim == 'inf' else t_lim
    template = Template(max_states)

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

        learner = Learner(template, batch_data, time_limit,
                          existing_model=global_model, mode='reasoning', debug=debug)

        new_model = learner.induce_iteratively(train_path, str(target_class),
                                               mini_batch_size, min_precision, global_model, max_local_iters)

        show_model_msg(new_model)

        selected_mini_batch = new_model.get_most_urgent_mini_batch(select_next_batch_by)

        if new_model.global_f1() > best_performance:
            best_performance = new_model.global_f1()
            global_model = new_model.automaton
