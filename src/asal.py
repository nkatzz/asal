import os
import sys

# Need to add the project root in the pythonpath.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.args_parser import parse_args

if "--help" in sys.argv or "-h" in sys.argv:
    parser = parse_args()
    parser.parse_args()
    sys.exit(0)

from statistics import mean
from src.asal.template import Template
from src.asal.auxils import get_train_data
from src.asal.asal import Asal
from src.asal.test_model_multproc import test_model_mproc
from src.logger import *
from src.args_parser import parse_args
from src.asal.asp import get_induction_program, get_test_program
from src.asal.auxils import f1

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    t_lim = args.tlim
    max_states = args.states  # 4
    target_class = args.tclass  # 2
    mini_batch_size = args.batch_size  # 10000
    shuffle = False
    selected_mini_batch = 0  # Could be randomized.
    mcts_iterations = args.mcts_iters  # 1
    expl_rate = args.exp_rate  # 0.005
    max_children = args.mcts_children  # 100  # 200  # 100
    path_scoring = False
    train_path = args.train
    test_path = args.test

    """
    train_path = os.path.normpath(
        os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
        dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'train.csv')

    test_path = os.path.normpath(
        os.getcwd() + os.sep + os.pardir + os.sep + 'data' + os.sep +
        dataset + os.sep + 'folds' + os.sep + fold + os.sep + 'test.csv')
    """

    if args.eval is not None:
        with open(args.eval, 'r') as file:
            sfa = file.read()
        result = test_model_mproc(sfa, args, path_scoring=False)
        tps, fps, fns = result.get_tps(), result.get_fps(), result.get_fns()
        logger.info(f'TPs, FPs, FNs: {tps}, {fps}, {fns}, '
                    f'F1-score: {f1(tps, fps, fns)}')
        sys.exit(-1)

    tmpl = Template(max_states, target_class)

    logger.debug(f'The induction program is:\n{get_induction_program(args, tmpl)}')

    train_data = get_train_data(train_path, str(target_class), mini_batch_size, shuffle=shuffle)
    seed_data = train_data[selected_mini_batch]

    # This learns something from the seed mini-batch.
    mcts = Asal(args, train_data, tmpl)

    if mcts_iterations > 1:  # revise from new batches.
        mcts.run_mcts()

    mode = "simple" if args.show == "s" else "reasoning"
    logger.info(blue(f'\nBest model found:\n{mcts.best_model.show(mode=mode)}\n\n'
                     f'F1-score on training set: {mcts.best_model.global_performance} '
                     f'(TPs, FPs, FNs: {mcts.best_model.global_performance_counts})\n'
                     f'Generated models: {mcts.generated_models_count}\n'
                     f'Average grounding time: {mean(mcts.grounding_times)}\n'
                     f'Average solving time: {mean(mcts.solving_times)}\n'
                     f'Model evaluation time: {sum(mcts.testing_times)}\n'
                     f'Total training time: {mcts.total_training_time}'))

    logger.info(yellow('Evaluating on testing set...'))
    logger.debug(f'The testing program is:\n{get_test_program(args)}')

    test_model_mproc(mcts.best_model, args, path_scoring=path_scoring)

    logger.info(yellow(f'On testing set: TPs, FPs, FNs: {mcts.best_model.global_performance_counts}, '
                       f'F1-score: {round(mcts.best_model.global_performance, 4)}'))
