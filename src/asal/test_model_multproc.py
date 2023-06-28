from src.asal.structs import Automaton, TestResult
from src.asal.auxils import get_train_data, f1
from src.asal.tester import Tester
from src.asal.tester_py import TesterPy
import multiprocessing as mp
import functools
import time
from src.asal.logger import *


def combine(res_1: TestResult, res_2: TestResult) -> TestResult:
    tps_seqs = res_1.tp_seqs + res_2.tp_seqs
    fps_seqs = res_1.fp_seqs + res_2.fp_seqs
    fn_seqs = res_1.fn_seqs + res_2.fn_seqs
    scores_per_batch = {**res_1.scores_per_batch, **res_2.scores_per_batch}
    scored_paths = res_1.scored_paths
    for path in res_2.scored_paths.keys():
        new_tps, new_fps = res_2.scored_paths[path]
        if path in scored_paths:
            prev_tps, prev_fps = scored_paths[path]
            scored_paths[path] = (prev_tps + new_tps, prev_fps + new_fps)
        else:
            scored_paths[path] = (new_tps, new_fps)

    return TestResult(tps_seqs, fps_seqs, fn_seqs, scored_paths, scores_per_batch)


def test_model_mproc(_automaton, data_path: str,
                     target_class: str, batch_size: int,
                     path_scoring=True, shuffle=False,
                     data_whole=None, test_with_clingo=True) -> TestResult:
    """Multiprocessing version of the test_model method in tester.py"""

    # automaton = _automaton.show(mode='reasoning') if isinstance(_automaton, Automaton) else _automaton
    automaton = _automaton
    logger.debug(f'Testing (on training set) for model:\n{automaton}')

    data = get_train_data(data_path, target_class,
                          batch_size, shuffle=shuffle) if data_whole is None else data_whole

    ds = [(key, data[key]) for key in data.keys()]

    # Use a partial function since Pool.map does not support lambda functions.
    # With the partial function below we achieve the same as:
    # pool.map(lambda batch: test_on_batch(batch, automaton, path_scoring), ds)
    # which won't compile.
    test_function = functools.partial(test_on_batch,
                                      target_class=target_class,
                                      automaton=automaton,
                                      path_scoring=path_scoring,
                                      test_with_clingo=test_with_clingo)

    start = time.time()

    with mp.Pool(processes=mp.cpu_count()) as pool:
        p = pool.map(test_function, ds)
        result = functools.reduce(combine, p)

    end = time.time()
    tps, fps, fns = result.get_tps(), result.get_fps(), result.get_fns()
    logger.debug(f' Testing time: {end - start} secs | TPs, FPs, FNs, F1: '
                 f'{tps}, {fps}, {fns}, {f1(tps, fps, fns)}')

    if isinstance(_automaton, Automaton):
        _automaton.global_performance_counts = (tps, fps, fns)
        _automaton.global_performance = f1(tps, fps, fns)
        _automaton.scored_paths = result.scored_paths
        _automaton.counts_per_batch = result.scores_per_batch
        _automaton.update_seqs_store((result.tp_seqs, result.fp_seqs, result.fn_seqs))

    return result


def test_on_batch(batch, target_class, automaton, path_scoring, test_with_clingo=True) -> TestResult:

    batch_id, batch_data = batch[0], batch[1]

    if test_with_clingo:
        automaton = automaton.show(mode='reasoning') if isinstance(automaton, Automaton) else automaton
        tester = Tester(batch_data, target_class, automaton, path_scoring=path_scoring)
        tester.test_model()
        tps, fps, fns = tester.tp_seq_ids, tester.fp_seq_ids, tester.fn_seq_ids
        # tps, fps, fns = len(tester.tp_seq_ids), len(tester.fp_seq_ids), len(tester.fn_seq_ids)
        paths = tester.scored_paths
        return TestResult(tps, fps, fns, paths, {batch_id: (len(tps), len(fps), len(fns))})
    else:
        test_on_batch_no_asp(batch_data, target_class, automaton, path_scoring)
        return TestResult(0, 0, 0, [], {batch_id: (0, 0, 0)})


def test_on_batch_no_asp(batch, target_class, automaton, path_scoring) -> TestResult:
    tester = TesterPy(batch, target_class, automaton, path_scoring)
    mvar_seqs = tester.convert_seqs()
