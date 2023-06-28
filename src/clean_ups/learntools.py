from src.asal.template import Template
from src.asal.learner import Learner
from src.asal.tester import test_model, rewrite_automaton
from src.asal.constraints import generate_constraints
import logging
from multiprocessing import Process, Queue
from src.asal.test_model_multproc import test_model_mproc
from src.asal.structs import ScoredModel, Automaton, TestResult


def call_learner(learner: Learner, q: Queue):
    result = learner.induce_models()
    q.put(result)


def call(learner: Learner, multiproc: bool):
    if multiproc:
        q = Queue()
        process = Process(target=call_learner, args=(learner, q))
        process.start()
        model = q.get()
        process.join()
    else:
        model = learner.induce_models()

    return model


def _test_model(model, train_path, target_class, mini_batch_size, multiproc):
    if not multiproc:
        scored_paths, seqs, counts_per_batch = test_model(model, train_path, str(target_class), mini_batch_size)
        result_object = TestResult(seqs[0], seqs[1],
                                   seqs[2], scored_paths, counts_per_batch)
    else:
        result_object = test_model_mproc(model, train_path, str(target_class), mini_batch_size)
    return result_object


def process_mini_batch(current_global_model: Automaton,
                       template: Template,
                       batch_data: str,
                       mini_batch_size,
                       train_path,
                       time_limit,
                       target_class,
                       max_iters: int,
                       min_precision,
                       prev_path_constraints: list[str],
                       multiproc=True,
                       debug=False):

    generated_models: list[ScoredModel] = []

    flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]

    for i in range(max_iters):

        """Collect all constraints (existing plus those generated so far in this mini-batch)."""
        constraints = prev_path_constraints + flat_map(lambda x: x.constraints, generated_models)

        if not constraints:
            constraints = None
        else:
            logging.debug(f' Constraints:\n{constraints}')

        learner = Learner(template, batch_data, time_limit,
                          existing_model=current_global_model, mode='reasoning',
                          constraints=constraints, debug=debug)

        model: Automaton = call(learner, multiproc)
        transformed_model, guards_map = rewrite_automaton(model)

        test_result = _test_model(transformed_model, train_path,
                                  str(target_class), mini_batch_size, multiproc)

        scored_paths = test_result.scored_paths
        global_counts = (test_result.get_tps(), test_result.get_fps(), test_result.get_fns())
        counts_per_batch = test_result.scores_per_batch

        model.update_seqs_store((test_result.tp_seqs, test_result.fp_seqs, test_result.fn_seqs))

        # new_constraints = generate_path_constraints(scored_paths, guards_map, min_precision)
        new_constraints = generate_constraints(scored_paths, guards_map, min_precision, current_global_model)
        scored_model = ScoredModel(model, new_constraints, global_counts, counts_per_batch)
        generated_models.append(scored_model)
        global_counts = (test_result.get_tps(), test_result.get_fps(), test_result.get_fns())

        if not new_constraints:  # then the last model scores above min_precision.
            break

    # best_model = max(generated_models, key=lambda x: x.global_f1)
    best_model = max(generated_models, key=lambda x: x.global_precision())
    generated_models.remove(best_model)  # Remove to avoid adding constraints related to best_model (see next line).
    constraints = list(set(prev_path_constraints + flat_map(lambda x: x.constraints, generated_models)))

    return best_model, constraints
